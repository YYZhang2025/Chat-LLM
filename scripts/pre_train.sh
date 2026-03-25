#!/bin/bash

source "$(dirname "$0")/env.sh"

export OMP_NUM_THREADS=1

NUM_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
DEPTH=24
TARGET_PARAM_DATA_RATIO=20
RUNNING_NAME="pre_train_${DEPTH}L_${TARGET_PARAM_DATA_RATIO}x"  # for logging and checkpoint naming


torchrun --standalone --nproc_per_node=$NUM_GPUS -m \
    trainer.pre_train -- \
    --running_name=$RUNNING_NAME \
    --depth=$DEPTH \
    --target_param_data_ratio=$TARGET_PARAM_DATA_RATIO \
    --device_batch_size=32 
