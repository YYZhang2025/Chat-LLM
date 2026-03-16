import os

import torch
import torch.distributed as dist


def is_ddp_requested() -> bool:
    return all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))


def is_ddp_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_dist_info():
    if is_ddp_requested():
        assert all(var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def clean_dist():
    if is_ddp_initialized():
        dist.destroy_process_group()


def dist_init(device_type="cuda"):  # cuda|cpu|mps
    """Basic initialization that we keep doing over and over, so make common."""

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), (
            "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
        )
    if device_type == "mps":
        assert torch.backends.mps.is_available(), (
            "Your PyTorch installation is not configured for MPS but device_type is 'mps'"
        )

    # Reproducibility
    # Note that we set the global seeds here, but most of the code uses explicit rng objects.
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)

    # Precision
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)  # mps|cpu

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device
