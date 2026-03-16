import os

import rich
import torch

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _detect_compute_dtype():
    env = os.environ.get("TRAINING_DTYPE")
    if env is not None:
        return (_DTYPE_MAP[env],)

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16

    return torch.float32


COMPUTE_DTYPE = _detect_compute_dtype()


def autodetect_device_type():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def format_with_commas(n: int) -> str:
    return f"{n:,}"


def print_master(s: str = "", type: str = "info", *args):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    if rank == 0:
        if type == "info":
            rich.print(f"[blue]{s}[/blue]", *args)
        elif type == "error":
            rich.print(f"[red]{s}[/red]", *args)
        elif type == "success":
            rich.print(f"[green]{s}[/green]", *args)
        else:
            rich.print(s, *args)


def print_dict_master(d: dict):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    if rank == 0:
        rich.console.Console().log(d, log_locals=True)


def get_base_dir():
    # co-locate chat-llm intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("CHAT_LLM_BASE_DIR"):
        nanochat_dir = os.environ.get("CHAT_LLM_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "chat-llm")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir
