import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
from dataclasses import asdict, dataclass
from functools import partial

import fire
import torch
import wandb

from chat_llm.dataloaders.pre_train import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from chat_llm.engine import GenerateEngine, sample_prompts
from chat_llm.eval.eval_common import evaluate_bpb
from chat_llm.eval.eval_core import evaluate_core
from chat_llm.model.attention import USE_FA3
from chat_llm.model.llm import build_model_meta, estimate_flops
from chat_llm.optim import optimizer_step, set_optimizer, update_optimizer_state
from chat_llm.scaling_law import (
    get_num_scaling_params,
    get_target_batch_size,
    get_target_tokens_num,
    get_target_weight_decay,
)
from chat_llm.tokenizer import get_token_bytes, get_tokenizer
from chat_llm.utils.checkpoint import save_checkpoint
from chat_llm.utils.common import (
    COMPUTE_DTYPE,
    autodetect_device_type,
    format_with_commas,
    get_peak_flops,
    print_master,
)
from chat_llm.utils.dist import clean_dist, dist_init


@dataclass
class Config:
    running_name: str = ""
    device_type: str = "cuda"

    # Model hyperparameters
    depth: int = 20
    aspect_ratio: int = 64  # use to determine model width
    head_dim: int = 128
    max_seq_len: int = 2048
    window_pattern: str = "SSSL"

    num_iterations: int = -1
    target_flops: float = -1.0
    target_param_data_ratio: float = 10.5  # tokens / parameter
    device_batch_size: int = 32
    total_batch_size: int = -1

    # Optimizer hyperparameters
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.008
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    weight_decay: float = 0.28

    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05

    # Checkpointing and evaluation
    resume_from_step: int = -1

    eval_every: int = 1000
    sample_every: int = 500
    save_every: int = -1

    eval_tokens: int = 80 * 524288
    core_metric_max_per_task: int = 500

    compiled: bool = True


prompts_sample = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]


def main(**kwargs):
    DATA_DIR = os.environ.get("DATA_DIR")
    TOKENIZER_DIR = os.environ.get("TOKENIZER_DIR")
    MODEL_DIR = os.environ.get("MODEL_DIR")
    assert DATA_DIR is not None, "DATA_DIR environment variable is not set"
    assert TOKENIZER_DIR is not None, "TOKENIZER_DIR environment variable is not set"
    assert MODEL_DIR is not None, "MODEL_DIR environment variable is not set"

    config = Config(**kwargs)

    if USE_FA3:
        print_master(
            "Using FA3 attention implementation.",
            type="warning",
        )
    else:
        print_master("Using standard attention implementation.")

    device_type = autodetect_device_type() if config.device_type == "" else config.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = dist_init(device_type)
    master_process = ddp_rank == 0  # master process will handle logging and checkpointing
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
    if device_type == "cuda":
        gpu_device_name = torch.cuda.get_device_name(0)
        peak_flops = get_peak_flops(gpu_device_name)
        print_master(f"Detected {gpu_device_name} device. Peak FLOPs: {format_with_commas(peak_flops)}")

    # WanDB setup (only on master process)
    if master_process:
        wandb_run = wandb.init(project="chat-llm", name=config.running_name, config=asdict(config))

    # Get Tokenizer
    tokenizer = get_tokenizer(TOKENIZER_DIR)
    vocab_size = tokenizer.get_vocab_size()
    print_master(f"Vocab size: {vocab_size:,}")

    # Build model
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
    print_master(
        f"Model built with {format_with_commas(sum(p.numel() for p in model.parameters()))} parameters."
    )
    orig_model = model
    compiled_model = torch.compile(model, dynamic=False) if config.compiled else model

    # Compute training setup based on scaling laws
    num_scaling_params = get_num_scaling_params(orig_model)
    print_master(f"Number of scaling parameters in the model: {format_with_commas(num_scaling_params)}")

    targets_tokens_nums = get_target_tokens_num(num_scaling_params, config.target_param_data_ratio)
    print_master(
        f"Target number of training tokens based on scaling laws: {format_with_commas(targets_tokens_nums)} with target param-data ratio of {config.target_param_data_ratio}"
    )
    d12_ref = build_model_meta(
        depth=12,
        aspect_ratio=config.aspect_ratio,
        head_dim=config.head_dim,
        vocab_size=vocab_size,
        max_seq_len=config.max_seq_len,
        window_pattern=config.window_pattern,
    )  # creates the model on meta device

    # compute-optimal d12 training horizon in tokens (measured empirically)
    D_REF = (config.target_param_data_ratio) * get_num_scaling_params(d12_ref)
    B_REF = 2**19  # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

    total_batch_size = (
        config.total_batch_size
        if config.total_batch_size > 0
        else get_target_batch_size(targets_tokens_nums, D_REF, B_REF)
    )
    print_master(f"Total batch size (tokens) for training: {format_with_commas(total_batch_size)} ")

    token_bytes = get_token_bytes(device=device)
    batch_lr_scale = 1.0
    batch_ratio = total_batch_size / B_REF

    if batch_ratio != 1.0:
        batch_lr_scale = batch_ratio**0.5  # η ∝ √(B/B_ref)

    weight_decay_scaled = get_target_weight_decay(
        config.weight_decay, total_batch_size, B_REF, D_REF, targets_tokens_nums
    )
    # tokens processed per forward+backward pass per GPU
    tokens_per_fwdbwd = config.device_batch_size * config.max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

    # Set Dataloader
    dataloader_resume_state_dict = None

    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        data_dir=DATA_DIR,  # data_dir
        tokenizer=tokenizer,
        B=config.device_batch_size,
        T=config.max_seq_len,
        split="train",
        device=device,
        resume_state_dict=dataloader_resume_state_dict,
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        data_dir=DATA_DIR,
        tokenizer=tokenizer,
        B=config.device_batch_size,
        T=config.max_seq_len,
        split="val",
        device=device,
    )

    # Start training loop
    num_flops_per_token = estimate_flops(orig_model)
    num_iterations = (
        config.num_iterations if config.num_iterations > 0 else targets_tokens_nums // total_batch_size
    )

    # Set Optimizer
    optimizer = set_optimizer(
        orig_model,
        un_embedding_lr=config.unembedding_lr * batch_lr_scale,
        embedding_lr=config.embedding_lr * batch_lr_scale,
        matrix_lr=config.matrix_lr * batch_lr_scale,
        weight_decay=weight_decay_scaled,
        scalar_lr=config.scalar_lr * batch_lr_scale,
    )

    optimizer_update = partial(
        update_optimizer_state,
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        warmdown_ratio=config.warmdown_ratio,
        num_iterations=num_iterations,
        final_lr_frac=config.final_lr_frac,
        weight_decay_scaled=weight_decay_scaled,
    )
    scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None

    print_master(f"Starting training for {format_with_commas(num_iterations)} iterations...")

    step = 0
    x, y, dataloader_state_dict = next(train_loader)
    torch.cuda.empty_cache() if device_type == "cuda" else None
    while True:
        last_step = step == num_iterations

        # Evaluation loss on the validation set (in bits per byte, bpb)
        if config.eval_every > 0 and (step + 1) % config.eval_every == 0:
            compiled_model.eval()
            val_loader = build_val_loader()
            eval_steps = config.eval_tokens // (
                config.device_batch_size * config.max_seq_len * ddp_world_size
            )
            print_master(f"Evaluating at step {step:,}... Eval steps: {format_with_commas(eval_steps)}")
            val_bpb = evaluate_bpb(compiled_model, val_loader, eval_steps, token_bytes)
            print_master(f"Step {step:,} | Validation bpb: {val_bpb:.4f}")

            if master_process:
                wandb_run.log(
                    {
                        "val/bpb": val_bpb,
                    },
                    step=step,
                )

            compiled_model.train()

        # Save checkpoint
        if master_process and config.save_every > 0 and (step + 1) % config.save_every == 0:
            save_checkpoint(
                MODEL_DIR,
                step,
                orig_model.state_dict(),  # model parameters
                optimizer.state_dict(),  # optimizer state
                {},
                rank=ddp_rank,
            )

            print_master(f"Checkpoint saved at step {step:,}")

        if master_process and config.sample_every > 0 and (step + 1) % config.sample_every == 0:
            compiled_model.eval()
            engine = GenerateEngine(orig_model, tokenizer)
            print_master("Sampling prompts...", type="info")
            results = sample_prompts(prompts_sample, engine)
            for i, generated in enumerate(results):
                print_master(f"Sample {i + 1}:", type="info")
                print_master(f"Generation: {generated}", type="info")

            compiled_model.train()

        # Training step
        if last_step:
            break

        synchronize()  # make sure all data is ready
        t0 = time.time()
        micro_losses = []
        for _ in range(grad_accum_steps):
            loss = compiled_model(x, y)
            train_loss = loss.item()
            micro_losses.append(train_loss)
            loss = loss / grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            x, y, dataloader_state_dict = next(train_loader)

        # -- Optimizer step
        optimizer_update(cur_step=step)
        optimizer_step(optimizer, scaler)
        orig_model.zero_grad(set_to_none=True)

        synchronize()
        t1 = time.time()
        dt = t1 - t0

        # Logging
        if master_process:
            train_loss = sum(micro_losses) / len(micro_losses)
            current_lr = optimizer.param_groups[0]["lr"]
            max_memory_bytes = get_max_memory()
            max_memory_gb = max_memory_bytes / (1024**3)
            tokens_this_step = total_batch_size
            tokens_seen = (step + 1) * total_batch_size
            throughput_tokens_per_sec = tokens_this_step / max(dt, 1e-12)

            flops_per_sec = num_flops_per_token * total_batch_size / dt
            mfu = 100 * flops_per_sec / (peak_flops * ddp_world_size)

            print_master(
                f"Step {step:,} | Loss: {train_loss:.4f} | LR: {current_lr:.2e}  | Time: {dt:.2f}s | Throughput: {format_with_commas(int(throughput_tokens_per_sec))} tokens/s | Max Memory: {max_memory_gb:.2f} GB | MFU: {mfu:.2f}%",
                type="info",
            )
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/lr": current_lr,
                    "train/step_time_sec": dt,
                    "train/max_memory_gb": max_memory_gb,
                    "train/throughput_tokens_per_sec": throughput_tokens_per_sec,
                    "train/tokens_seen": tokens_seen,
                    "train/grad_accum_steps": grad_accum_steps,
                    "train/batch_lr_scale": batch_lr_scale,
                    "train/weight_decay_scaled": weight_decay_scaled,
                    "train/mfu": mfu,
                },
                step=step,
            )

        step += 1
    # ---------- End of training loop ----------

    # Evaluate final model on CORE metric\
    results = {}
    compiled_model.eval()
    results = evaluate_core(orig_model, tokenizer, device, config.core_metric_max_per_task)
    print_master("CORE evaluation results:")
    if master_process:
        print("\n=== CORE Results ===")
        print(f"CORE Metric: {results['core_metric']:.4f}\n")

        print("Per-task accuracy:")
        for task, acc in sorted(results["results"].items()):
            print(f"  {task:30s}  acc={acc:.4f}")

        wandb_run.log(results)

    print_master(f"Checkpoint saved at step {step:,}")
    # Cleanup
    if master_process:
        # save results and final checkpoint
        with open(os.path.join(MODEL_DIR, "final_results.json"), "w") as f:
            json.dump(results, f)

        save_checkpoint(
            MODEL_DIR,
            step,
            orig_model.state_dict(),  # model parameters
            optimizer.state_dict(),  # optimizer state
            {},
            rank=ddp_rank,
        )
        wandb_run.finish()
    if ddp:
        clean_dist()


if __name__ == "__main__":
    fire.Fire(main)
