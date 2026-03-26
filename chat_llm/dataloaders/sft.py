from typing import Callable, Optional

import torch


def sft_data_generator_bos_bestfit(
    *,
    split: str,
    train_dataset,
    val_dataset,
    tokenizer,
    config,
    ddp_rank: int,
    ddp_world_size: int,
    device,
    device_type: str,
    progress_state: Optional[dict] = None,
):
    """
    BOS-aligned dataloader for SFT with bestfit-pad packing.

    Each row in the batch starts with BOS (beginning of a conversation).
    Conversations are packed using best-fit algorithm. When no conversation fits,
    the row is padded (instead of cropping) to ensure no tokens are ever discarded.
    Padding positions have targets masked with -1 (ignore_index for cross-entropy).

    config:
        split: "train" or "val"
        train_dataset: training dataset
        val_dataset: validation dataset
        tokenizer: tokenizer object, must provide:
            - get_bos_token_id()
            - render_conversation(conversation) -> (ids, mask)
        config: config object, must provide:
            - max_seq_len
            - device_batch_size
            - num_iterations
        ddp_rank: distributed rank
        ddp_world_size: distributed world size
        device: torch device
        device_type: "cuda" or "cpu"
        progress_state: optional mutable dict for exposing training status.
            Example:
            {
                "last_step": False,
                "approx_progress": 0.0,
                "current_epoch": 1,
            }

    Yields:
        inputs:  [B, T] int32
        targets: [B, T] int64, masked with -1 where loss should be ignored
    """
    assert split in {"train", "val"}, "split must be 'train' or 'val'"

    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0

    row_capacity = config.max_seq_len + 1  # +1 for shifted target
    bos_token = tokenizer.get_bos_token_id()

    if progress_state is None:
        progress_state = {
            "last_step": False,
            "approx_progress": 0.0,
            "current_epoch": 1,
        }

    conv_buffer = []
    cursor = ddp_rank
    consumed = ddp_rank
    epoch = 1
    it = 0

    def refill_buffer(buffer_size: int):
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conversation)
            conv_buffer.append((ids, mask))

            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1

    buffer_size = 100

    while True:
        rows = []
        mask_rows = []
        row_lengths = []

        for _ in range(config.device_batch_size):
            row = []
            mask_row = []
            padded = False
            content_len = 0

            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer(buffer_size)

                remaining = row_capacity - len(row)

                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += ddp_world_size
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    padded = True
                    break

            row_lengths.append(content_len if padded else row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        it += 1

        if split == "train":
            if 0 < config.num_iterations <= it:
                progress_state["last_step"] = True

            progress_state["current_epoch"] = epoch
            if config.num_iterations > 0:
                progress_state["approx_progress"] = it / config.num_iterations
            else:
                progress_state["approx_progress"] = consumed / dataset_size

            if consumed >= dataset_size:
                progress_state["last_step"] = True

        use_cuda = device_type == "cuda"

        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()

        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:].to(device=device)
        targets[mask_targets == 0] = -1

        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len - 1 :] = -1

        yield inputs, targets


def build_sft_dataloader(
    *,
    train_dataset,
    val_dataset,
    tokenizer,
    config,
    ddp_rank: int,
    ddp_world_size: int,
    device,
    device_type: str,
    progress_state: Optional[dict] = None,
) -> tuple[Callable[[], object], Callable[[], object]]:
    """
    Returns:
        build_train_loader, build_val_loader
    Each callable returns a fresh generator.
    """

    def build_train_loader():
        return sft_data_generator_bos_bestfit(
            split="train",
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            config=config,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            device=device,
            device_type=device_type,
            progress_state=progress_state,
        )

    def build_val_loader():
        return sft_data_generator_bos_bestfit(
            split="val",
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            config=config,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            device=device,
            device_type=device_type,
            progress_state=progress_state,
        )

    return build_train_loader, build_val_loader
