import copy
import itertools
import math
import os
from dataclasses import asdict, dataclass

import fire
import torch
import torch.distributed as dist
import wandb

from chat_llm.engine import GenerateEngine
from chat_llm.task import GSM8K
from chat_llm.utils.checkpoint import load_checkpoint, save_checkpoint
from chat_llm.utils.common import autodetect_device_type, get_base_dir, print_master
from chat_llm.utils.dist import clean_dist, dist_init

# -----------------------------------------------------------------------------
# Config


@dataclass
class Config:
    running_name: str = "dummy"
    device_type: str = "cuda"

    # Model hyperparameters
    depth: int = 26
    aspect_ratio: int = 64
    head_dim: int = 128
    max_seq_len: int = 2048
    window_pattern: str = "SSSL"

    num_iterations: int = -1
    target_flops: float = -1.0
    target_param_data_ratio: float = 10.5
    device_batch_size: int = 8
    total_batch_size: int = -1

    # Optimizer hyperparameters
    embedding_lr: float = 0.2
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    weight_decay: float = 0.0

    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    init_lr_frac: float = 0.05

    # Checkpointing and evaluation
    resume_from_step: int = -1
    model_step: int = -1

    eval_every: int = 60
    sample_every: int = 1000
    save_every: int = 60
    chatcore_every: int = 1000
    chatcore_max_cat: int = -1
    chatcore_max_sample: int = 24

    eval_tokens: int = 80 * 524288
    core_metric_max_per_task: int = 500

    # SFT
    mmlu_epochs: int = 2
    gsm8k_epochs: int = 1

    compiled: bool = True

    # RL / GRPO-specific
    examples_per_step: int = 16
    num_samples: int = 8
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50

    clip_eps: float = 0.2
    kl_beta: float = 0.02
    grpo_updates: int = 4
    adv_eps: float = 1e-4

    # optional model loading tag
    model_tag: str = ""


# -----------------------------------------------------------------------------
# Small helpers


@dataclass
class RolloutBatch:
    sequences_all: list
    inputs_all: torch.Tensor  # (B, T)
    targets_all: torch.Tensor  # (B, T)
    gen_mask_all: torch.Tensor  # (B, T)
    rewards_all: torch.Tensor  # (B,)
    advantages_all: torch.Tensor  # (B,)
    old_logp_all: torch.Tensor  # (B, T)
    ref_logp_all: torch.Tensor  # (B, T)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = mask.sum().clamp(min=eps)
    return (x * mask).sum() / denom


def approx_kl_estimator(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """
    Schulman positive KL estimator often used in GRPO/PPO implementations:
        KL ~= exp(ref_logp - logp) - (ref_logp - logp) - 1
    """
    delta = ref_logp - logp
    return torch.exp(delta) - delta - 1.0


def gather_token_logprobs_from_loss(model, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Assumes:
        model(inputs, targets, loss_reduction="none")
    returns token NLLs that can be reshaped to (B, T).
    """
    nll = model(inputs, targets, loss_reduction="none").view_as(inputs)
    return -nll


def get_lr_multiplier(step: int, num_steps: int, cfg: Config) -> float:
    """
    Warmup + cosine warmdown style scheduler, matching your config style more closely.
    """
    if num_steps <= 1:
        return 1.0

    progress = step / max(num_steps - 1, 1)

    if cfg.warmup_ratio > 0 and progress < cfg.warmup_ratio:
        warmup_progress = progress / cfg.warmup_ratio
        return cfg.init_lr_frac + (1.0 - cfg.init_lr_frac) * warmup_progress

    warmdown_start = 1.0 - cfg.warmdown_ratio
    if progress < warmdown_start:
        return 1.0

    warmdown_progress = (progress - warmdown_start) / max(cfg.warmdown_ratio, 1e-8)
    cosine = 0.5 * (1.0 + math.cos(math.pi * warmdown_progress))
    return cfg.final_lr_frac + (1.0 - cfg.final_lr_frac) * cosine


def run_gsm8k_eval(
    task,
    tokenizer,
    engine,
    ddp_rank: int,
    ddp_world_size: int,
    device,
    num_samples: int = 1,
    max_examples: int | None = None,
    max_completion_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int = 50,
):
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)

    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        generated_token_sequences, masks = engine.generate_batch(
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
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({"is_correct": is_correct})

        yield {
            "idx": idx,
            "outcomes": outcomes,
        }


def load_rl_model(cfg: Config, device):
    """
    You imported load_checkpoint, so here we wrap it into a model/tokenizer/meta loader.
    Adjust this if your project has a different builder API.
    """
    ckpt = load_checkpoint(
        checkpoint_dir=MODEI_WEIGHTS_DIR,
        step=cfg.model_step,
        device=device,
        load_optimizer=False,
    )

    model = ckpt["model"]
    tokenizer = ckpt["tokenizer"]
    meta = ckpt.get("meta", {})

    return model, tokenizer, meta


# -----------------------------------------------------------------------------
# Main


def main(**kwargs):
    cfg = Config(**kwargs)

    # -------------------------------------------------------------------------
    # Init distributed / device
    device_type = autodetect_device_type() if cfg.device_type == "" else cfg.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = dist_init(device_type)
    master_process = ddp_rank == 0

    # -------------------------------------------------------------------------
    # wandb
    if master_process:
        wandb_run = wandb.init(
            project="chat-llm-grpo",
            name=cfg.running_name,
            config=asdict(cfg),
        )

    # -------------------------------------------------------------------------
    # Load trainable model
    model, tokenizer, meta = load_rl_model(cfg, device)
    engine = GenerateEngine(model, tokenizer)

    # Load frozen reference model
    ref_model, _, _ = load_rl_model(cfg, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # -------------------------------------------------------------------------
    # Tasks
    train_task = GSM8K(subset="main", split="train")
    val_task = GSM8K(subset="main", split="test")

    num_steps = (len(train_task) // cfg.examples_per_step) * cfg.gsm8k_epochs
    if cfg.num_iterations > 0:
        num_steps = cfg.num_iterations

    print_master(f"Calculated number of steps: {num_steps}")

    # -------------------------------------------------------------------------
    # Optimizer
    optimizer = model.setup_optimizer(
        unembedding_lr=cfg.unembedding_lr,
        embedding_lr=cfg.embedding_lr,
        matrix_lr=cfg.matrix_lr,
        weight_decay=cfg.weight_decay,
    )

    for group in optimizer.param_groups:
        base_lr = group["lr"]
        group["lr"] = base_lr * cfg.init_lr_frac
        group["initial_lr"] = base_lr

    # -------------------------------------------------------------------------
    # Batch geometry
    print_master(f"Total sampled sequences per step: {cfg.examples_per_step * cfg.num_samples}")

    assert cfg.examples_per_step % ddp_world_size == 0, "examples_per_step must be divisible by world size"
    examples_per_rank = cfg.examples_per_step // ddp_world_size
    print_master(f"Examples per rank: {examples_per_rank}")

    # -------------------------------------------------------------------------
    # Rollout generator
    @torch.no_grad()
    def get_batch(current_step: int):
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        rank_indices = range(ddp_rank, len(train_task), ddp_world_size)

        for example_idx in itertools.cycle(rank_indices):
            conversation = train_task[example_idx]

            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)

            model.eval()
            generated_token_sequences = []
            masks = []

            assert cfg.num_samples % cfg.device_batch_size == 0, (
                "num_samples must be divisible by device_batch_size"
            )
            num_sampling_passes = cfg.num_samples // cfg.device_batch_size

            for sampling_pass in range(num_sampling_passes):
                seed = hash((current_step, example_idx, sampling_pass, ddp_rank)) & 0x7FFFFFFF
                seqs_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=cfg.device_batch_size,
                    max_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    seed=seed,
                )
                generated_token_sequences.extend(seqs_batch)
                masks.extend(masks_batch)

            rewards = []
            for sample_tokens in generated_token_sequences:
                generated_tokens = sample_tokens[prefix_length:]
                generated_text = tokenizer.decode(generated_tokens)
                reward = train_task.reward(conversation, generated_text)
                rewards.append(float(reward))

            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

            mu = rewards.mean()
            sigma = rewards.std(unbiased=False)
            advantages = (rewards - mu) / (sigma + cfg.adv_eps)

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

            old_logp = gather_token_logprobs_from_loss(model, inputs, targets_for_logp).detach()
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

    # -------------------------------------------------------------------------
    # Training loop
    for step in range(num_steps):
        batch_iterator = get_batch(step)

        # ---------------------------------------------------------------------
        # Eval
        if step % cfg.eval_every == 0:
            model.eval()

            eval_num_samples = min(cfg.device_batch_size, cfg.num_samples)
            passk = torch.zeros(eval_num_samples, device=device)

            records = list(
                run_gsm8k_eval(
                    val_task,
                    tokenizer,
                    engine,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                    device=device,
                    num_samples=eval_num_samples,
                    max_examples=cfg.core_metric_max_per_task,
                    max_completion_tokens=cfg.max_new_tokens,
                    temperature=1.0,
                    top_k=cfg.top_k,
                )
            )

            for k in range(1, eval_num_samples + 1):
                passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)

            num_records = torch.tensor(len(records), dtype=torch.long, device=device)
            if ddp:
                dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
                dist.all_reduce(passk, op=dist.ReduceOp.SUM)

            passk = passk / max(num_records.item(), 1)
            print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, eval_num_samples + 1)]
            print_master(f"Step {step} | {', '.join(print_passk)}")

            wandb_run.log(
                {"step": step, **{f"pass@{k}": passk[k - 1].item() for k in range(1, eval_num_samples + 1)}}
            )

        # ---------------------------------------------------------------------
        # Collect rollout groups
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
        wandb_run.log(
            {
                "step": step,
                "reward": mean_reward,
                "sequence_length": mean_sequence_length,
            }
        )

        # ---------------------------------------------------------------------
        # GRPO updates
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss_value = 0.0
        total_pg_value = 0.0
        total_kl_value = 0.0
        num_loss_terms = 0

        for update_idx in range(cfg.grpo_updates):
            for example_idx, batch in enumerate(rollout_groups):
                B = batch.inputs_all.size(0)
                assert B % cfg.device_batch_size == 0
                num_microbatches = B // cfg.device_batch_size

                for mb_idx in range(num_microbatches):
                    b0 = mb_idx * cfg.device_batch_size
                    b1 = (mb_idx + 1) * cfg.device_batch_size

                    inputs = batch.inputs_all[b0:b1]
                    targets = batch.targets_all[b0:b1]
                    gen_mask = batch.gen_mask_all[b0:b1]
                    advantages = batch.advantages_all[b0:b1]
                    old_logp = batch.old_logp_all[b0:b1]
                    ref_logp = batch.ref_logp_all[b0:b1]

                    logp = gather_token_logprobs_from_loss(model, inputs, targets)

                    ratio = torch.exp(logp - old_logp)

                    adv = advantages.unsqueeze(-1).expand_as(logp)

                    unclipped = ratio * adv
                    clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
                    pg_obj_token = torch.minimum(unclipped, clipped)

                    kl_token = approx_kl_estimator(logp, ref_logp)

                    valid_mask = gen_mask
                    pg_obj = masked_mean(pg_obj_token, valid_mask)
                    kl_mean = masked_mean(kl_token, valid_mask)

                    loss = -(pg_obj - cfg.kl_beta * kl_mean)

                    scale = cfg.grpo_updates * examples_per_rank * num_microbatches
                    loss = loss / scale
                    loss.backward()

                    total_loss_value += loss.item() * scale
                    total_pg_value += pg_obj.item()
                    total_kl_value += kl_mean.item()
                    num_loss_terms += 1

                    print_master(
                        f"Step {step}/{num_steps} | update {update_idx + 1}/{cfg.grpo_updates} | "
                        f"group {example_idx + 1}/{examples_per_rank} | "
                        f"microbatch {mb_idx + 1}/{num_microbatches} | "
                        f"pg_obj={pg_obj.item():.6f} | kl={kl_mean.item():.6f} | loss={loss.item():.6f}"
                    )

        # ---------------------------------------------------------------------
        # Step optimizer
        lrm = get_lr_multiplier(step, num_steps, cfg)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm

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

        wandb_run.log(
            {
                "step": step,
                "pg_obj": mean_pg,
                "kl": mean_kl,
                "loss": mean_loss,
                "lrm": lrm,
            }
        )

        # ---------------------------------------------------------------------
        # Save
        if master_process and ((step > 0 and step % cfg.save_every == 0) or step == num_steps - 1):
            base_dir = get_base_dir()
            depth = model.config.n_layer
            output_dirname = cfg.model_tag if cfg.model_tag else f"d{depth}"
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

    if master_process:
        wandb_run.finish()

    clean_dist()


if __name__ == "__main__":
    fire.Fire(main)
