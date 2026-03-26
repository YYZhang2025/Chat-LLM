import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import fire
import torch
import torch.distributed as dist
import wandb

from chat_llm.dataloaders.sft import build_sft_dataloader
from chat_llm.eval.eval_common import evaluate_bpb
from chat_llm.model.attention import USE_FA3
from chat_llm.model.llm import build_model_meta
from chat_llm.optim import set_optimizer
from chat_llm.task import ARC, GSM8K, MMLU, SmolTalk, TaskMixture
from chat_llm.tokenizer import get_token_bytes, get_tokenizer
from chat_llm.utils.checkpoint import load_checkpoint, save_checkpoint
from chat_llm.utils.common import (
    COMPUTE_DTYPE,
    autodetect_device_type,
    format_with_commas,
    get_peak_flops,
    print_master,
)
from chat_llm.utils.dist import clean_dist, dist_init

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


@dataclass
class Config:
    running_name: str = ""
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
    device_batch_size: int = 32
    total_batch_size: int = -1

    # Optimizer hyperparameters
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    weight_decay: float = 0.28

    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    init_lr_frac: float = 0.1

    # Checkpointing and evaluation
    resume_from_step: int = -1
    model_step: int = -1

    eval_every: int = 1000
    sample_every: int = 500
    save_every: int = 1000

    eval_tokens: int = 80 * 524288
    core_metric_max_per_task: int = 500

    # SFT
    mmlu_epochs: int = 2
    gsm8k_epochs: int = 2

    compiled: bool = True


def is_ddp_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_grad_accum_steps(config: Config, ddp_world_size: int) -> int:
    """
    total_batch_size is interpreted as total tokens per optimizer step:
        total_batch_size = micro_bsz * seq_len * world_size * grad_accum_steps
    """
    if config.total_batch_size <= 0:
        return 1

    denom = config.device_batch_size * config.max_seq_len * ddp_world_size
    assert denom > 0
    assert config.total_batch_size % denom == 0, (
        f"total_batch_size ({config.total_batch_size}) must be divisible by "
        f"device_batch_size * max_seq_len * ddp_world_size ({denom})"
    )
    return config.total_batch_size // denom


def get_lr_multiplier(
    step: int, total_steps: int, warmup_steps: int, warmdown_ratio: float, final_lr_frac: float
) -> float:
    """
    LR schedule:
    1. linear warmup
    2. cosine decay
    3. final lr floor = final_lr_frac
    """
    if total_steps <= 0:
        return 1.0

    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)

    decay_start = warmup_steps
    decay_end = total_steps
    if step >= decay_end:
        return final_lr_frac

    progress = (step - decay_start) / max(1, decay_end - decay_start)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_lr_frac + (1.0 - final_lr_frac) * cosine


def get_muon_momentum(step: int) -> float:
    # 先给一个稳定默认值；如果你项目里有自己的 muon schedule，可以替换这里
    return 0.95


def main(**kwargs):
    DATA_DIR = os.environ.get("DATA_DIR")
    TOKENIZER_DIR = os.environ.get("TOKENIZER_DIR")
    MODEL_DIR = os.environ.get("MODEL_DIR")

    assert DATA_DIR is not None, "DATA_DIR environment variable is not set"
    assert TOKENIZER_DIR is not None, "TOKENIZER_DIR environment variable is not set"
    assert MODEL_DIR is not None, "MODEL_DIR environment variable is not set"

    config = Config(**kwargs)

    if USE_FA3:
        print_master("Using FA3 attention implementation.", type="warning")
    else:
        print_master("Using standard attention implementation.")

    device_type = autodetect_device_type() if config.device_type == "" else config.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = dist_init(device_type)
    master_process = ddp_rank == 0

    synchronize = torch.cuda.synchronize if device_type == "cuda" else (lambda: None)
    get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else (lambda: 0)

    if device_type == "cuda":
        torch.cuda.set_device(device)
        gpu_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        peak_flops = get_peak_flops(gpu_device_name)
        print_master(f"Detected {gpu_device_name} device. Peak FLOPs: {format_with_commas(peak_flops)}")
    else:
        peak_flops = None

    wandb_run = None
    if master_process:
        wandb_run = wandb.init(project="chat-llm", name=config.running_name, config=asdict(config))

    # Tokenizer
    tokenizer = get_tokenizer(TOKENIZER_DIR)
    vocab_size = tokenizer.get_vocab_size()
    token_bytes = get_token_bytes()
    print_master(f"Vocab size: {vocab_size:,}")

    # Model
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

    # Load checkpoint
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir=MODEL_DIR,
        step=config.model_step,
        device=device,
        load_optimizer=True,
        rank=ddp_rank,
    )

    if model_data is not None:
        model.load_state_dict(model_data)

    original_model = model
    compiled_model = torch.compile(model) if config.compiled else model

    optimizer = set_optimizer(
        model=model,
        embedding_lr=config.embedding_lr,
        unembedding_lr=config.unembedding_lr,
        matrix_lr=config.matrix_lr,
        scalar_lr=config.scalar_lr,
        weight_decay=config.weight_decay,
    )

    scaler = (
        torch.amp.GradScaler("cuda") if (device_type == "cuda" and COMPUTE_DTYPE == torch.float16) else None
    )

    # Save each param group's base lr for scheduler
    for group in optimizer.param_groups:
        base_lr = group["lr"]
        group["base_lr"] = base_lr
        group["lr"] = base_lr * config.init_lr_frac

    # Dataset
    train_tasks = [
        SmolTalk(split="train"),
        *[MMLU(subset="auxiliary_train", split="train") for _ in range(config.mmlu_epochs)],
        *[GSM8K(subset="main", split="train") for _ in range(config.gsm8k_epochs)],
    ]
    train_dataset = TaskMixture(train_tasks)

    val_dataset = TaskMixture(
        [
            SmolTalk(split="val"),
            MMLU(subset="auxiliary_train", split="val"),
            GSM8K(subset="main", split="val"),
        ]
    )

    progress_state = {
        "last_step": False,
        "approx_progress": 0.0,
        "current_epoch": 1,
    }

    build_train_loader, build_val_loader = build_sft_dataloader(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        config=config,  # 如果你的 build_sft_dataloader 用的是 config，就把这里改回 config=config
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        device=device,
        device_type=device_type,
        progress_state=progress_state,
    )

    train_loader = build_train_loader()
    val_loader = build_val_loader()

    grad_accum_steps = get_grad_accum_steps(config, ddp_world_size)
    print_master(f"grad_accum_steps: {grad_accum_steps}")

    step = 0
    x, y = next(train_loader)

    compiled_model.train()

    try:
        while True:
            if ddp:
                last_step_tensor = torch.tensor(
                    int(progress_state["last_step"]),
                    dtype=torch.int32,
                    device=device,
                )
                dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
                last_step = last_step_tensor.item() == 1
            else:
                last_step = progress_state["last_step"]

            # Eval
            if last_step or (config.eval_every > 0 and (step + 1) % config.eval_every == 0):
                compiled_model.eval()
                eval_steps = max(1, config.eval_tokens // (config.device_batch_size * config.max_seq_len))
                val_bpb = evaluate_bpb(
                    compiled_model,
                    val_loader,
                    eval_steps,
                    token_bytes=token_bytes,
                )
                print_master(f"Step {step + 1}: Validation Bits-Per-Byte: {val_bpb:.4f}")
                compiled_model.train()

            if last_step:
                print_master("Stopping: reached last_step.")
                break

            synchronize()
            t0 = time.time()

            optimizer.zero_grad(set_to_none=True)

            train_loss = None
            for _ in range(grad_accum_steps):
                loss = compiled_model(x, y)
                train_loss = loss.detach()
                loss = loss / grad_accum_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                x, y = next(train_loader)

            # LR update
            total_steps_for_sched = config.num_iterations if config.num_iterations > 0 else max(step + 1, 1)
            lrm = get_lr_multiplier(
                step=step,
                total_steps=total_steps_for_sched,
                warmup_steps=config.warmup_steps,
                warmdown_ratio=config.warmdown_ratio,
                final_lr_frac=config.final_lr_frac,
            )
            muon_momentum = get_muon_momentum(step)

            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * lrm
                if group.get("kind") == "muon":
                    group["momentum"] = muon_momentum

            # Optimizer step
            if scaler is not None:
                scaler.unscale_(optimizer)
                if is_ddp_initialized():
                    found_inf = getattr(scaler, "_found_inf_per_device", None)
                    if found_inf is not None:
                        vals = scaler._found_inf_per_device(optimizer)
                        for v in vals.values():
                            dist.all_reduce(v, op=dist.ReduceOp.MAX)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            compiled_model.zero_grad(set_to_none=True)

            synchronize()
            t1 = time.time()
            step_time = t1 - t0
            step += 1

            # Logging
            tokens_per_step = (
                config.device_batch_size * config.max_seq_len * ddp_world_size * grad_accum_steps
            )
            throughput = tokens_per_step / max(step_time, 1e-8)

            if master_process:
                log_dict = {
                    "train/loss": train_loss.item() if train_loss is not None else None,
                    "train/lr_multiplier": lrm,
                    "train/epoch": progress_state["current_epoch"],
                    "train/approx_progress": progress_state["approx_progress"],
                    "perf/step_time_sec": step_time,
                    "perf/tokens_per_sec": throughput,
                    "perf/max_memory_bytes": get_max_memory(),
                    "step": step,
                }
                wandb.log(log_dict, step=step)
            print_master(
                f"Step {step} | "
                f"Loss: {train_loss.item():.4f} | "
                f"LR Mult: {lrm:.4f} | "
                f"Time: {step_time:.2f}s | "
                f"Throughput: {throughput:,.0f} tok/s | "
                f"Max Memory: {get_max_memory() / (1024**3):.2f} GB | "
                f"Epoch: {progress_state['current_epoch']} | "
                f"Progress: {progress_state['approx_progress']:.4f}"
            )
            # Save
            if config.save_every > 0 and step % config.save_every == 0:
                save_checkpoint(
                    checkpoint_dir=MODEL_DIR,
                    step=step,
                    model_data=original_model.state_dict(),
                    optimizer_data=optimizer.state_dict(),
                    meta_data={
                        "step": step,
                        "config": asdict(config),
                        "progress_state": progress_state,
                    },
                    rank=ddp_rank,
                )

            # Clean
            if step % 5000 == 0:
                gc.collect()
                if device_type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        if master_process and wandb_run is not None:
            wandb_run.finish()
        clean_dist()


if __name__ == "__main__":
    fire.Fire(main)
