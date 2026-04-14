import copy
import itertools
import os
from dataclasses import asdict, dataclass

import fire
import torch
import torch.distributed as dist
import wandb

from chat_llm.engine import GenerateEngine
from chat_llm.model.attention import USE_FA3
from chat_llm.model.llm import build_model_meta
from chat_llm.optim import set_optimizer
from chat_llm.task import GSM8K
from chat_llm.tokenizer import get_tokenizer
from chat_llm.utils.checkpoint import save_checkpoint
from chat_llm.utils.common import autodetect_device_type, get_base_dir, print_master
from chat_llm.utils.dist import clean_dist, dist_init

SYSTEM_PROMPT = """You are a careful reasoning assistant.

Before answering, reason through the task internally in a structured, step-by-step way. Check for mistakes, edge cases, and consistency. Keep this reasoning internal.

Then provide only the final answer in the following format:

answer: #### <final numerical answer>

Additional rules:
- Never output anything before `answer:`
- Do not expose internal chain-of-thought
- The final numerical answer must appear after `####`
- Example: answer: #### 42
- Be precise, direct, and correct
"""


@dataclass
class Config:
    # Runtime
    running_name: str = ""
    device_type: str = "cuda"
    compile: bool = True

    # Model hyperparameters
    depth: int = 26
    aspect_ratio: int = 64
    head_dim: int = 128
    max_seq_len: int = 2048
    window_pattern: str = "SSSL"

    # Model loading
    model_tag: str | None = None
    model_step: int = -1

    # Training horizon
    num_epochs: int = 1

    # Batching / rollout
    device_batch_size: int = 8
    examples_per_step: int = 16
    num_samples: int = 8

    # Generation
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50

    # GRPO objective
    clip_eps: float = 0.2
    kl_beta: float = 0.02
    grpo_updates: int = 4
    adv_eps: float = 1e-4

    # Optimizer
    embedding_lr: float = 0.005
    unembedding_lr: float = 0.0002
    matrix_lr: float = 0.001
    scalar_lr: float = 0.01
    weight_decay: float = 0.1

    # LR schedule
    warmup_ratio: float = 0.03
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    init_lr_frac: float = 0.1

    # Eval / checkpoint
    eval_every: int = 1000
    eval_examples: int = 400
    save_every: int = 1000


@dataclass
class RolloutBatch:
    sequences_all: list
    inputs_all: torch.Tensor
    targets_all: torch.Tensor
    gen_mask_all: torch.Tensor
    rewards_all: torch.Tensor
    advantages_all: torch.Tensor
    old_logp_all: torch.Tensor
    ref_logp_all: torch.Tensor


def apply_system_prompt(conversation: dict) -> dict:
    convo = copy.deepcopy(conversation)
    messages = convo["messages"]

    if len(messages) > 0 and messages[0]["role"] == "system":
        messages[0]["content"] = SYSTEM_PROMPT
    else:
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    return convo


def normalize_for_scoring(text: str) -> str:
    """
    Keep only the visible answer portion while preserving the GSM8K #### marker.

    Examples:
    - "answer: #### 42" -> "#### 42"
    - "answer:\n#### 42" -> "#### 42"
    - "#### 42" -> "#### 42"
    - anything else -> stripped raw text
    """
    stripped = text.strip()
    lower = stripped.lower()

    if lower.startswith("answer:"):
        stripped = stripped[len("answer:") :].strip()

    return stripped


def gather_token_logprobs_from_loss(forward_model, inputs, targets):
    nll = forward_model(inputs, targets, loss_reduction="none").view_as(inputs)
    return -nll


def masked_mean(x, mask, eps=1e-8):
    denom = mask.sum().clamp(min=eps)
    return (x * mask).sum() / denom


def approx_kl_estimator(logp, ref_logp):
    delta = ref_logp - logp
    return torch.exp(delta) - delta - 1.0


def get_lr_multiplier(it, num_steps, config: Config):
    if num_steps <= 1:
        return 1.0

    warmup_steps = int(config.warmup_ratio * num_steps)
    warmdown_steps = int(config.warmdown_ratio * num_steps)
    warmdown_start = max(warmup_steps, num_steps - warmdown_steps)

    if warmup_steps > 0 and it < warmup_steps:
        alpha = it / warmup_steps
        return config.init_lr_frac + alpha * (1.0 - config.init_lr_frac)

    if it < warmdown_start:
        return 1.0

    decay_steps = num_steps - warmdown_start
    if decay_steps <= 0:
        return config.final_lr_frac

    alpha = (it - warmdown_start) / decay_steps
    return 1.0 - alpha * (1.0 - config.final_lr_frac)


def main(**kwargs):
    tokenizer_dir = os.environ.get("TOKENIZER_DIR")
    model_dir = os.environ.get("MODEL_DIR")

    assert tokenizer_dir is not None, "TOKENIZER_DIR environment variable is not set"
    assert model_dir is not None, "MODEL_DIR environment variable is not set"

    config = Config(**kwargs)
    user_config = asdict(config)

    if USE_FA3:
        print_master("Using FA3 attention implementation.", type="warning")
    else:
        print_master("Using standard attention implementation.")

    device_type = autodetect_device_type() if config.device_type == "" else config.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = dist_init(device_type)
    master_process = ddp_rank == 0

    if config.running_name and config.running_name != "dummy":
        wandb_run = wandb.init(project="nanochat-grpo", name=config.running_name, config=user_config)
    else:
        wandb_run = None

    tokenizer = get_tokenizer(tokenizer_dir)
    vocab_size = tokenizer.get_vocab_size()
    print_master(f"Vocab size: {vocab_size:,}")

    model = build_model_meta(
        depth=config.depth,
        aspect_ratio=config.aspect_ratio,
        head_dim=config.head_dim,
        vocab_size=vocab_size,
        max_seq_len=config.max_seq_len,
        window_pattern=config.window_pattern,
    )
    model.to_empty(device=device)
    model.init_weights()

    step_to_load = config.model_step
    if step_to_load == -1:
        model_files = [f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".pt")]
        if len(model_files) == 0:
            raise ValueError(f"No checkpoint files found in {model_dir}")
        model_steps = [int(f[len("model_") : -len(".pt")]) for f in model_files]
        step_to_load = max(model_steps)
        print_master(f"Auto-detected latest checkpoint step: {step_to_load}")

    model_path = os.path.join(model_dir, f"model_{step_to_load:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    if model_data is not None:
        model.load_state_dict(model_data)

    compiled_model = torch.compile(model) if config.compile else model

    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    engine = GenerateEngine(model, tokenizer)

    train_task = GSM8K(subset="main", split="train")
    val_task = GSM8K(subset="main", split="test")

    num_steps = (len(train_task) // config.examples_per_step) * config.num_epochs
    print_master(f"Calculated number of steps: {num_steps}")

    @torch.no_grad()
    def get_batch(train_step):
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        rank_indices = range(ddp_rank, len(train_task), ddp_world_size)

        for example_idx in itertools.cycle(rank_indices):
            base_conversation = train_task[example_idx]
            conversation = apply_system_prompt(base_conversation)

            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)

            model.eval()
            generated_token_sequences = []
            masks = []

            assert config.num_samples % config.device_batch_size == 0, (
                "num_samples must be divisible by device_batch_size"
            )
            num_sampling_passes = config.num_samples // config.device_batch_size

            for sampling_pass in range(num_sampling_passes):
                seed = hash((train_step, example_idx, sampling_pass, ddp_rank)) & 0x7FFFFFFF
                seqs_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=config.device_batch_size,
                    max_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    seed=seed,
                )
                generated_token_sequences.extend(seqs_batch)
                masks.extend(masks_batch)

            rewards = []
            for sample_tokens in generated_token_sequences:
                generated_tokens = sample_tokens[prefix_length:]
                generated_text = tokenizer.decode(generated_tokens)
                scored_text = normalize_for_scoring(generated_text)
                reward = train_task.reward(base_conversation, scored_text)
                rewards.append(float(reward))

            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

            mu = rewards.mean()
            sigma = rewards.std(unbiased=False)
            advantages = (rewards - mu) / (sigma + config.adv_eps)

            max_length = max(len(seq) for seq in generated_token_sequences)
            padded_sequences = [
                seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences
            ]
            padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]

            ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)
            mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)

            inputs = ids[:, :-1].contiguous()
            targets = ids[:, 1:].clone().contiguous()
            gen_mask = mask_ids[:, 1:].to(torch.float32)

            targets_for_logp = targets.clone()
            targets_for_logp[gen_mask == 0] = -1

            old_logp = gather_token_logprobs_from_loss(compiled_model, inputs, targets_for_logp).detach()
            ref_logp = gather_token_logprobs_from_loss(ref_model, inputs, targets_for_logp).detach()

            yield RolloutBatch(
                sequences_all=generated_token_sequences,
                inputs_all=inputs,
                targets_all=targets_for_logp,
                gen_mask_all=gen_mask,
                rewards_all=rewards,
                advantages_all=advantages,
                old_logp_all=old_logp,
                ref_logp_all=ref_logp,
            )

    def run_gsm8k_eval(
        task,
        tokenizer,
        engine,
        max_examples=None,
        num_samples=1,
        max_completion_tokens=256,
        temperature=0.0,
        top_k=50,
    ):
        max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)

        for idx in range(ddp_rank, max_examples, ddp_world_size):
            base_conversation = task[idx]
            conversation = apply_system_prompt(base_conversation)

            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)

            assert num_samples <= config.device_batch_size, "eval num_samples must fit device batch size"
            generated_token_sequences, _ = engine.generate_batch(
                tokens,
                num_samples=num_samples,
                max_tokens=max_completion_tokens,
                temperature=temperature,
                top_k=top_k,
            )

            outcomes = []
            for sample_tokens in generated_token_sequences:
                generated_tokens = sample_tokens[prefix_length:]
                generated_text = tokenizer.decode(generated_tokens)
                scored_text = normalize_for_scoring(generated_text)
                is_correct = task.evaluate(base_conversation, scored_text)
                outcomes.append({"is_correct": is_correct, "raw_text": generated_text})

            yield {"idx": idx, "outcomes": outcomes}

    optimizer = set_optimizer(
        model=model,
        embedding_lr=config.embedding_lr,
        unembedding_lr=config.unembedding_lr,
        matrix_lr=config.matrix_lr,
        scalar_lr=config.scalar_lr,
        weight_decay=config.weight_decay,
    )

    for group in optimizer.param_groups:
        base_lr = group["lr"]
        group["base_lr"] = base_lr
        group["lr"] = base_lr * config.init_lr_frac

    print_master(f"Total sampled sequences per step: {config.examples_per_step * config.num_samples}")

    assert config.examples_per_step % ddp_world_size == 0, "examples_per_step must be divisible by world size"
    examples_per_rank = config.examples_per_step // ddp_world_size
    print_master(f"Examples per rank: {examples_per_rank}")

    for step in range(num_steps):
        batch_iterator = get_batch(step)

        if step % config.eval_every == 0:
            model.eval()

            passk = torch.zeros(config.device_batch_size, device=device)
            records = list(
                run_gsm8k_eval(
                    val_task,
                    tokenizer,
                    engine,
                    num_samples=config.device_batch_size,
                    max_examples=config.eval_examples,
                    temperature=1.0,
                    top_k=config.top_k,
                )
            )

            for k in range(1, config.device_batch_size + 1):
                passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)

            num_records = torch.tensor(len(records), dtype=torch.long, device=device)
            if ddp:
                dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
                dist.all_reduce(passk, op=dist.ReduceOp.SUM)

            passk = passk / max(num_records.item(), 1)
            accuracy = passk[0].item()

            print_passk = [
                f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, config.device_batch_size + 1)
            ]
            print_master(f"Step {step} | Accuracy: {accuracy:.4f} | {', '.join(print_passk)}")

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "step": step,
                        "accuracy": accuracy,
                        **{f"pass@{k}": passk[k - 1].item() for k in range(1, config.device_batch_size + 1)},
                    }
                )

        rollout_groups = []
        rewards_list = []
        sequence_lengths = []

        for _ in range(examples_per_rank):
            batch = next(batch_iterator)
            rollout_groups.append(batch)
            rewards_list.append(batch.rewards_all.mean().item())
            sequence_lengths.extend(len(seq) for seq in batch.sequences_all)

        mean_reward = sum(rewards_list) / max(len(rewards_list), 1)
        mean_sequence_length = sum(sequence_lengths) / max(len(sequence_lengths), 1)

        if ddp:
            mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float32, device=device)
            mean_seq_tensor = torch.tensor(mean_sequence_length, dtype=torch.float32, device=device)
            dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(mean_seq_tensor, op=dist.ReduceOp.AVG)
            mean_reward = mean_reward_tensor.item()
            mean_sequence_length = mean_seq_tensor.item()

        print_master(
            f"Step {step}/{num_steps} | rollout reward: {mean_reward:.4f} | "
            f"avg seq len: {mean_sequence_length:.2f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "step": step,
                    "reward": mean_reward,
                    "sequence_length": mean_sequence_length,
                }
            )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss_value = 0.0
        total_pg_value = 0.0
        total_kl_value = 0.0
        num_loss_terms = 0

        for update_idx in range(config.grpo_updates):
            for example_idx, batch in enumerate(rollout_groups):
                B = batch.inputs_all.size(0)
                assert B % config.device_batch_size == 0
                num_microbatches = B // config.device_batch_size

                for mb_idx in range(num_microbatches):
                    b0 = mb_idx * config.device_batch_size
                    b1 = (mb_idx + 1) * config.device_batch_size

                    inputs = batch.inputs_all[b0:b1]
                    targets = batch.targets_all[b0:b1]
                    gen_mask = batch.gen_mask_all[b0:b1]
                    advantages = batch.advantages_all[b0:b1]
                    old_logp = batch.old_logp_all[b0:b1]
                    ref_logp = batch.ref_logp_all[b0:b1]

                    logp = gather_token_logprobs_from_loss(compiled_model, inputs, targets)

                    ratio = torch.exp(logp - old_logp)
                    adv = advantages.unsqueeze(-1).expand_as(logp)

                    unclipped = ratio * adv
                    clipped = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv
                    pg_obj_token = torch.minimum(unclipped, clipped)

                    kl_token = approx_kl_estimator(logp, ref_logp)
                    valid_mask = gen_mask

                    pg_obj = masked_mean(pg_obj_token, valid_mask)
                    kl_mean = masked_mean(kl_token, valid_mask)

                    loss = -(pg_obj - config.kl_beta * kl_mean)

                    scale = config.grpo_updates * examples_per_rank * num_microbatches
                    loss = loss / scale
                    loss.backward()

                    total_loss_value += loss.item() * scale
                    total_pg_value += pg_obj.item()
                    total_kl_value += kl_mean.item()
                    num_loss_terms += 1

                    print_master(
                        f"Step {step}/{num_steps} | update {update_idx + 1}/{config.grpo_updates} | "
                        f"group {example_idx + 1}/{examples_per_rank} | microbatch {mb_idx + 1}/{num_microbatches} | "
                        f"pg_obj={pg_obj.item():.6f} | kl={kl_mean.item():.6f} | loss={loss.item():.6f}"
                    )

        lrm = get_lr_multiplier(step, num_steps, config)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * lrm

        optimizer.step()
        model.zero_grad(set_to_none=True)

        mean_pg = total_pg_value / max(num_loss_terms, 1)
        mean_kl = total_kl_value / max(num_loss_terms, 1)
        mean_loss = total_loss_value / max(num_loss_terms, 1)

        if ddp:
            stats = torch.tensor([mean_pg, mean_kl, mean_loss], dtype=torch.float32, device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.AVG)
            mean_pg, mean_kl, mean_loss = stats.tolist()

        print_master(
            f"Step {step}/{num_steps} | mean_pg={mean_pg:.6f} | mean_kl={mean_kl:.6f} | "
            f"mean_loss={mean_loss:.6f} | lrm={lrm:.6f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "step": step,
                    "pg_obj": mean_pg,
                    "kl": mean_kl,
                    "loss": mean_loss,
                    "lrm": lrm,
                }
            )

        if master_process and ((step > 0 and step % config.save_every == 0) or step == num_steps - 1):
            base_dir = get_base_dir()
            output_dirname = config.model_tag if config.model_tag else f"d{config.depth}"
            checkpoint_dir = os.path.join(base_dir, "chatrl_grpo_checkpoints", output_dirname)

            model_config_kwargs = model.config.__dict__

            save_checkpoint(
                checkpoint_dir,
                step,
                model.state_dict(),
                None,
                {
                    "model_config": model_config_kwargs,
                },
            )
            print_master(f"✅ Saved model checkpoint to {checkpoint_dir}")

    if master_process and wandb_run is not None and hasattr(wandb_run, "finish"):
        wandb_run.finish()

    clean_dist()


if __name__ == "__main__":
    fire.Fire(main)
