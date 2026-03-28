import argparse
import copy
import itertools
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import wandb

from chat_llm.engine import GenerateEngine
from chat_llm.task import GSM8K
from chat_llm.utils.checkpoint import load_checkpoint, save_checkpoint
from chat_llm.utils.common import autodetect_device_type, get_base_dir, print_master
from chat_llm.utils.dist import clean_dist, dist_init, get_dist_info

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="DeepSeekMath-style GRPO on GSM8K")

# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")

# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")

# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")

# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K")

# Batching / rollout
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument(
    "--examples-per-step", type=int, default=16, help="total questions per optimization step across all ranks"
)
parser.add_argument("--num-samples", type=int, default=8, help="group size: samples per question")

# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="max generated tokens")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")

# GRPO / PPO-style objective
parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO/GRPO clip epsilon")
parser.add_argument("--kl-beta", type=float, default=0.02, help="KL coefficient vs reference policy")
parser.add_argument(
    "--grpo-updates",
    type=int,
    default=4,
    help="number of optimization passes on the same sampled rollout group",
)
parser.add_argument(
    "--adv-eps", type=float, default=1e-4, help="epsilon added to reward std in z-score normalization"
)

# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="embedding LR (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="unembedding LR (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="matrix LR (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR fraction")

# Eval / ckpt
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="eval question count")
parser.add_argument("--save-every", type=int, default=60, help="save every N steps")

args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# Init compute
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = dist_init(device_type)
master_process = ddp_rank == 0

# -----------------------------------------------------------------------------
# wandb
wandb_run = wandb.init(project="nanochat-grpo", name=args.run, config=user_config)


# -----------------------------------------------------------------------------
# Load policy model (trainable)
model, tokenizer, meta = load_model(
    "sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step
)
engine = GenerateEngine(model, tokenizer)

# Load frozen reference model (same initialization as policy start)
ref_model, _, _ = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)

# -----------------------------------------------------------------------------
# Tasks
train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")

num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print_master(f"Calculated number of steps: {num_steps}")


# -----------------------------------------------------------------------------
# Helper data structure
@dataclass
class RolloutBatch:
    sequences_all: list
    inputs_all: torch.Tensor  # (B, T)
    targets_all: torch.Tensor  # (B, T)
    gen_mask_all: torch.Tensor  # (B, T), 1 on generated positions used in loss
    rewards_all: torch.Tensor  # (B,)
    advantages_all: torch.Tensor  # (B,)
    old_logp_all: torch.Tensor  # (B, T) token log probs under old policy
    ref_logp_all: torch.Tensor  # (B, T) token log probs under reference model


# -----------------------------------------------------------------------------
# Utilities


def gather_token_logprobs_from_loss(model, inputs, targets):
    """
    Reuses nanochat-style per-token NLL path:
      model(inputs, targets, loss_reduction="none") -> flat or shaped NLL
    Returns token logprobs for the selected targets, shape (B, T)
    """
    nll = model(inputs, targets, loss_reduction="none").view_as(inputs)  # (B, T)
    return -nll


def masked_mean(x, mask, eps=1e-8):
    denom = mask.sum().clamp(min=eps)
    return (x * mask).sum() / denom


def approx_kl_estimator(logp, ref_logp):
    """
    DeepSeekMath cites Schulman's positive KL estimator.
    Common tokenwise form used in GRPO implementations:

        KL ~= exp(ref_logp - logp) - (ref_logp - logp) - 1

    where logp is current policy logprob and ref_logp is reference-policy logprob.
    """
    delta = ref_logp - logp
    return torch.exp(delta) - delta - 1.0


# -----------------------------------------------------------------------------
# Rollout generator


@torch.no_grad()
def get_batch():
    """
    Yields one question-group at a time for the current rank.

    For each question:
    - sample G responses from current/old policy
    - compute scalar rewards
    - z-score within the group => advantages
    - cache old logprobs and ref logprobs tokenwise
    """
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)

    for example_idx in itertools.cycle(rank_indices):
        conversation = train_task[example_idx]

        # Prompt tokens: keep assistant prefix, let model complete
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Sample group outputs from old policy = current policy snapshot before update
        model.eval()
        generated_token_sequences = []
        masks = []

        assert args.num_samples % args.device_batch_size == 0, (
            "num_samples must be divisible by device_batch_size"
        )
        num_sampling_passes = args.num_samples // args.device_batch_size

        for sampling_pass in range(num_sampling_passes):
            seed = hash((step, example_idx, sampling_pass, ddp_rank)) & 0x7FFFFFFF
            seqs_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=args.device_batch_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed,
            )
            generated_token_sequences.extend(seqs_batch)
            masks.extend(masks_batch)

        # Scalar rewards per sampled completion
        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            reward = train_task.reward(conversation, generated_text)
            rewards.append(float(reward))

        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        # Group-relative z-score normalization (DeepSeekMath style)
        mu = rewards.mean()
        sigma = rewards.std(unbiased=False)
        advantages = (rewards - mu) / (sigma + args.adv_eps)

        # Pad sequences to common time length
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_sequences = [
            seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences
        ]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]

        ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)  # (B, L)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)  # (B, L)

        # Standard causal LM shift
        inputs = ids[:, :-1].contiguous()
        targets = ids[:, 1:].clone().contiguous()

        # mask_ids[:, 1:] corresponds to targets positions
        gen_mask = mask_ids[:, 1:].to(torch.float32)  # 1 on generated tokens to optimize

        # Ignore non-generated positions in CE/NLL computation
        targets_for_logp = targets.clone()
        targets_for_logp[gen_mask == 0] = -1

        # Cache old logprobs and reference logprobs tokenwise
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


# -----------------------------------------------------------------------------
# Evaluation


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
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        assert num_samples <= args.device_batch_size, "eval num_samples must fit device batch size"
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


# -----------------------------------------------------------------------------
# Optimizer

optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

for group in optimizer.param_groups:
    base_lr = group["lr"]
    group["lr"] = base_lr * args.init_lr_frac
    group["initial_lr"] = base_lr


def get_lr_multiplier(it):
    return max(0.0, 1.0 - it / max(num_steps, 1))


# -----------------------------------------------------------------------------
# Batch geometry
print_master(f"Total sampled sequences per step: {args.examples_per_step * args.num_samples}")

assert args.examples_per_step % ddp_world_size == 0, "examples_per_step must be divisible by world size"
examples_per_rank = args.examples_per_step // ddp_world_size
print_master(f"Examples per rank: {examples_per_rank}")

batch_iterator = get_batch()

# -----------------------------------------------------------------------------
# Training loop
for step in range(num_steps):
    # ---------------------------------------------------------
    # Eval
    if step % args.eval_every == 0:
        model.eval()

        passk = torch.zeros(args.device_batch_size, device=device)
        records = list(
            run_gsm8k_eval(
                val_task,
                tokenizer,
                engine,
                num_samples=args.device_batch_size,
                max_examples=args.eval_examples,
                temperature=1.0,
                top_k=args.top_k,
            )
        )

        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)

        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)

        passk = passk / max(num_records.item(), 1)
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        print_master(f"Step {step} | {', '.join(print_passk)}")

        wandb_run.log(
            {"step": step, **{f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}}
        )

    # ---------------------------------------------------------
    # Collect rollout groups for this step
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

    # ---------------------------------------------------------
    # Multiple GRPO updates on same rollout groups
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss_value = 0.0
    total_pg_value = 0.0
    total_kl_value = 0.0
    num_loss_terms = 0

    for update_idx in range(args.grpo_updates):
        for example_idx, batch in enumerate(rollout_groups):
            B = batch.inputs_all.size(0)
            assert B % args.device_batch_size == 0
            num_microbatches = B // args.device_batch_size

            for mb_idx in range(num_microbatches):
                b0 = mb_idx * args.device_batch_size
                b1 = (mb_idx + 1) * args.device_batch_size

                inputs = batch.inputs_all[b0:b1]
                targets = batch.targets_all[b0:b1]
                gen_mask = batch.gen_mask_all[b0:b1]
                advantages = batch.advantages_all[b0:b1]
                old_logp = batch.old_logp_all[b0:b1]
                ref_logp = batch.ref_logp_all[b0:b1]

                # Current token logprobs
                logp = gather_token_logprobs_from_loss(model, inputs, targets)  # (b, T)

                # ratio = pi_theta / pi_old on sampled tokens
                ratio = torch.exp(logp - old_logp)

                # Outcome supervision: same scalar advantage for all generated tokens in one sample
                adv = advantages.unsqueeze(-1).expand_as(logp)

                # PPO/GRPO clipped surrogate
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
                pg_obj_token = torch.minimum(unclipped, clipped)

                # Tokenwise KL to reference model
                kl_token = approx_kl_estimator(logp, ref_logp)

                # Only generated tokens count
                valid_mask = gen_mask

                pg_obj = masked_mean(pg_obj_token, valid_mask)
                kl_mean = masked_mean(kl_token, valid_mask)

                # Maximize surrogate - beta * KL  <=> minimize negative
                loss = -(pg_obj - args.kl_beta * kl_mean)

                # Normalize across:
                # - examples_per_rank rollout groups
                # - grpo_updates repeated updates
                # - microbatches inside each group
                scale = args.grpo_updates * examples_per_rank * num_microbatches
                loss = loss / scale

                loss.backward()

                total_loss_value += loss.item() * scale
                total_pg_value += pg_obj.item()
                total_kl_value += kl_mean.item()
                num_loss_terms += 1

                print_master(
                    f"Step {step}/{num_steps} | update {update_idx + 1}/{args.grpo_updates} | "
                    f"group {example_idx + 1}/{examples_per_rank} | microbatch {mb_idx + 1}/{num_microbatches} | "
                    f"pg_obj={pg_obj.item():.6f} | kl={kl_mean.item():.6f} | loss={loss.item():.6f}"
                )

    # ---------------------------------------------------------
    # LR schedule + step
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    optimizer.step()
    model.zero_grad(set_to_none=True)

    mean_pg = total_pg_value / max(num_loss_terms, 1)
    mean_kl = total_kl_value / max(num_loss_terms, 1)
    mean_loss = total_loss_value / max(num_loss_terms, 1)

    # DDP aggregate logging
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

    # ---------------------------------------------------------
    # Save
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "chatrl_grpo_checkpoints", output_dirname)

        model_config_kwargs = model.config.__dict__

        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None,  # optimizer state omitted intentionally
            {
                "model_config": model_config_kwargs,
            },
        )
        print_master(f"✅ Saved model checkpoint to {checkpoint_dir}")

# -----------------------------------------------------------------------------
# Cleanup
if master_process and hasattr(wandb_run, "finish"):
    wandb_run.finish()

compute_cleanup()
