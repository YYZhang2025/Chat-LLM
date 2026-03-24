#!/bin/bash

source "$(dirname "$0")/env.sh"

export OMP_NUM_THREADS=1

NUM_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
RUNNING_NAME="pre_train_24L_9.5x"  # for logging and checkpoint naming


torchrun --standalone --nproc_per_node=$NUM_GPUS -m \
    trainer.pre_train -- \
    --running_name=$RUNNING_NAME \
    --depth=24 \
    --target-param-data-ratio=9.5 \
    --device-batch-size=32 \
    --num_iterations=200 

# evaluate the model: CORE metric, BPB on train/val, and draw samples
# torchrun --standalone --nproc_per_node=8 -m \
#     trainer.base_eval -- \
#     --device-batch-size=16
