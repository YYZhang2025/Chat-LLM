#!/bin/bash

export WANDB_API_KEY="your_wandb_api_key_here"
export CHATLLM_DATA_DIR="$HOME/path/to/your/data"


torchrun --standalone --nproc_per_node=8 -m \\
    trainer.pre_train -- \\
    --depth=24 \\
    --target-param-data-ratio=9.5 \\
    --device-batch-size=16 \\
    --run=$WANDB_RUN

# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=8 -m \\
    trainer.base_eval -- \\
    --device-batch-size=16
