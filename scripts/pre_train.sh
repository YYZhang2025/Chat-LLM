#!/bin/bash

source "$(dirname "$0")/env.sh"

export OMP_NUM_THREADS=1


torchrun --standalone --nproc_per_node=2 -m \
    trainer.pre_train -- \
    --depth=24 \
    --target-param-data-ratio=9.5 \
    --device-batch-size=16 \
    --run=$WANDB_RUN

# evaluate the model: CORE metric, BPB on train/val, and draw samples
# torchrun --standalone --nproc_per_node=8 -m \
#     trainer.base_eval -- \
#     --device-batch-size=16
