import os

from chat_llm.scaling_law import get_num_scaling_params

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from chat_llm.optim import optimizer_step, set_optimizer, update_optimizer_state
from chat_llm.tokenizer import get_tokenizer

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
from dataclasses import asdict, dataclass
from functools import partial

import fire
import torch
import wandb

from chat_llm.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from chat_llm.model.llm import LLMModel, ModelConfig
from chat_llm.utils.common import COMPUTE_DTYPE, autodetect_device_type, format_with_commas, print_master
from chat_llm.utils.dist import clean_dist, dist_init


@dataclass
class Config:
    run: str = "dummy"
    device_type: str = "cuda"
    # Model hyperparameters
    depth: int = 20
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
    unembedding_lr: float = 0.008
    weight_decay: float = 0.28
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05

    # Checkpointing and evaluation
    resume_from_step: int = -1
    eval_every: int = 250
    eval_tokens: int = 80 * 524288
    core_metric_every: int = 2000
    core_metric_max_per_task: int = 500
    sample_every: int = 2000
    save_every: int = -1

    compiled: bool = True


def build_model_meta(depth, aspect_ratio, head_dim, vocab_size, max_seq_len, window_pattern):
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    config = ModelConfig(
        embed_dim=model_dim,
        n_q_heads=num_heads,
        n_kv_heads=num_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_ff=4 * model_dim,
        window_pattern=window_pattern,
    )
    with torch.device("meta"):
        model_meta = LLMModel(config)
    return model_meta


def main(**kwargs):
    DATA_DIR = os.environ.get("DATA_DIR")
    TOKENIZER_DIR = os.environ.get("TOKENIZER_DIR")
    MODEL_DIR = os.environ.get("MODEL_DIR")
    assert DATA_DIR is not None, "DATA_DIR environment variable is not set"
    assert TOKENIZER_DIR is not None, "TOKENIZER_DIR environment variable is not set"
    assert MODEL_DIR is not None, "MODEL_DIR environment variable is not set"

    config = Config(**kwargs)

    device_type = autodetect_device_type() if config.device_type == "" else config.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = dist_init(device_type)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

    # WanDB setup (only on master process)
    if master_process:
        wandb_run = wandb.init(project="chat-llm", name=config.run, config=asdict(config))

    # Get Tokenizer
    tokenizer = get_tokenizer(TOKENIZER_DIR)
    vocab_size = tokenizer.get_vocab_size()
    print_master(f"Vocab size: {vocab_size:,}")

    # Build model on CPU first to avoid any GPU memory fragmentation issues. We will move it to the target device later.
    model = build_model_meta(
        depth=config.depth,
        aspect_ratio=config.aspect_ratio,
        head_dim=config.head_dim,
        vocab_size=vocab_size,
        max_seq_len=config.max_seq_len,
        window_pattern=config.window_pattern,
    )
    model.to_empty(
        device=device
    )  # 2) All tensors get storage on target device but with uninitialized (garbage) data

    model.init_weights()
    print_master(
        f"Model built with {format_with_commas(sum(p.numel() for p in model.parameters()))} parameters."
    )

    orig_model = model  # This  will point to the same model object throughout, even if we later wrap it in DDP or compile it, which makes it easier to save checkpoints without worrying about unwrapping or tracking multiple references to the model object.
    compiled_model = torch.compile(model, dynamic=False) if config.compiled else model
    
    
    
    from chat_llm.scaling_law import get_num_scaling_params, get_target_batch_size, get_target_tokens_num, get_target_weight_decay

    num_scaling_params = get_num_scaling_params(orig_model)
    tokens_per_fwdbwd = config.device_batch_size * config.max_seq_len
    targets_tokens_nums = get_target_tokens_num(num_scaling_params, config.target_param_data_ratio)
    d12_ref = build_model_meta(
        depth=12,
        aspect_ratio=config.aspect_ratio,
        head_dim=config.head_dim,
        vocab_size=vocab_size,
        max_seq_len=config.max_seq_len,
        window_pattern=config.window_pattern,
    )  # creates the model on meta device
    
    D_REF = (
        config.target_param_data_ratio
    ) get_num_scaling_params(d12_ref)  # compute-optimal d12 training horizon in tokens (measured empirically)
    B_REF = 2**19  # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

    total_batch_size = config.total_batch_size if config.total_batch_size > 0 else get_target_batch_size(
        num_scaling_params, targets_tokens_nums, D_REF, B_REF
    )
    
    batch_lr_scale = 1.0
    batch_ratio = total_batch_size / B_REF 
    
    if batch_ratio!= 1.0:
        batch_lr_scale = batch_ratio**0.5  # η ∝ √(B/B_ref)
    
    weight_decay_scaled = get_target_weight_decay(config.weight_decay, total_batch_size, B_REF, D_REF, targets_tokens_nums)
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    grad_accum_steps = config.total_batch_size // world_tokens_per_fwdbwd


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
        num_iterations=config.num_iterations,
        final_lr_frac=config.final_lr_frac,
        weight_decay_scaled=weight_decay_scaled,
    )
    scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None

    # Set Dataloader
    dataloader_resume_state_dict = None

    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        DATA_DIR,  # data_dir
        tokenizer,
        config.device_batch_size,
        config.max_seq_len,
        split="train",
        device=device,
        resume_state_dict=dataloader_resume_state_dict,
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, config.device_batch_size, config.max_seq_len, split="val", device=device
    )
    x, y, dataloader_state_dict = next(train_loader)  # kick off load of the very first batch of data

    # Start training loop

    step = 0
    tokens_per_fwdbwd = config.device_batch_size * config.max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    grad_accum_steps = config.total_batch_size // world_tokens_per_fwdbwd
    
    print_master(f"Tokens / micro-batch / rank: {config.device_batch_size} x {config.max_seq_len} = {tokens_per_fwdbwd:,}")
    print_master(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
    print_master(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

    from chat_llm.utils.common import print_dict_master
    print_dict_master({
        "Tokens per micro-batch per rank": tokens_per_fwdbwd,
        "Tokens per micro-batch": world_tokens_per_fwdbwd,
        "Total batch size": total_batch_size, 
        "Gradient accumulation steps": grad_accum_steps,  
    })
    while True:
        last_step = step == config.num_iterations

        # Evaluation
        # pass

        # Training step
        if last_step:
            break

        synchronize()  # make sure all data is ready
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            loss = compiled_model(x, y)
            train_loss = loss.item()
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
            current_lr = optimizer.param_groups[0]["lr"]

            print_master(
                f"Step {step:,} | Loss: {train_loss:.4f} | LR: {current_lr:.2e}  | Time: {dt:.2f}s | Max Mem: {format_with_commas(get_max_memory())} bytes"
            )
            # wandb.log(
            #     {
            #         "train/loss": train_loss,
            #         "train/lr": current_lr,
            #         "train/step_time": dt,
            #         "train/max_memory_bytes": get_max_memory(),
            #     },
            #     step=step,
            # )

        step += 1

    # Cleanup
    wandb_run.finish()
    if ddp:
        clean_dist()


if __name__ == "__main__":
    fire.Fire(main)
