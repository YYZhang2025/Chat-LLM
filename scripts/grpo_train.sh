#!/bin/bash

source "$(dirname "$0")/env.sh"

export OMP_NUM_THREADS=1

NUM_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
echo "Detected $NUM_GPUS GPUs for GRPO."

DEPTH=26
RUNNING_NAME="grpo_${DEPTH}L"  # for logging and checkpoint naming
DEVICE_BATCH_SIZE=16 # Adjust based on GPU memory;

torchrun --standalone --nproc_per_node=$NUM_GPUS -m \
    trainer.grpo_train -- \
    --running_name=$RUNNING_NAME \
    --depth=$DEPTH \
    --device_batch_size=$DEVICE_BATCH_SIZE 
