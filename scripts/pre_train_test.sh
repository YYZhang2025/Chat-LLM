#!/bin/bash

source "$(dirname "$0")/env.sh"

export OMP_NUM_THREADS=1

NUM_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
RUNNING_NAME="pre_train_24L_9.5x"  # for logging and checkpoint naming


torchrun --standalone --nproc_per_node=2 -m \
    trainer.pre_train -- \
    --running_name=$RUNNING_NAME \
    --aspect_ratio=16 \
    --max_seq_len=128 \
    --device_batch_size=2 \
    --num_iterations=3 \
    --eval_every=2 \
    --eval_tokens=1024 \
    --sample_every=2 \
    --core_metric_max_per_task=5 \
    --compiled=True \


