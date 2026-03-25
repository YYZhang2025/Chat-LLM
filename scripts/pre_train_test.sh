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
    trainer.pre_train \
    --running_name=$RUNNING_NAME \
    --depth=24 \
    --target_param_data_ratio=9.5 \
    --device_batch_size=32 \
    --num_iterations=100 \
    --eval_every=50 \
    --sample_every=20 \
    --core_metric_every=50 


# evaluate the model: CORE metric, BPB on train/val, and draw samples
# torchrun --standalone --nproc_per_node=8 -m \
#     trainer.base_eval -- \
#     --device-batch-size=16
