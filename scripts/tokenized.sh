#!/bin/bash
export CHATLLM_DATA_DIR="$HOME/path/to/your/data"


python -m download -n 8

python -m download -n 10 & DATASET_DOWNLOAD_PID=$!

python -m trainer.train_tokenizer

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID