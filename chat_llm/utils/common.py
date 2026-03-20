import os
import urllib.request

import rich
import torch
from filelock import FileLock
from rich.console import Console
from rich.table import Table

console = Console()

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


def print_master(s: str = "", type: str = "info", **kwargs):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    if rank == 0:
        if type == "info":
            rich.print(f"[blue][info] {s}[/blue]", **kwargs)
        elif type == "error":
            rich.print(f"[red][error] {s}[/red]", **kwargs)
        elif type == "success":
            rich.print(f"[green][success] {s}[/green]", **kwargs)
        else:
            rich.print(s, **kwargs)


def print_dict_master(d: dict, title: str = "Training Config"):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    if rank == 0:
        table = Table(title=title)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for k, v in d.items():
            table.add_row(str(k), str(v))

        console.log(table)


def get_base_dir():
    # co-locate chat-llm intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("CHAT_LLM_BASE_DIR"):
        base_dir = os.environ.get("CHAT_LLM_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        base_dir = os.path.join(cache_dir, "chat-llm")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()  # bytes

        # Write to local file
        with open(file_path, "wb") as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)


def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    # Table order matters: more specific patterns first.
    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere data center
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        # NVIDIA Ada data center
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA accelerators
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # Consumer RTX
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    return float("inf")
