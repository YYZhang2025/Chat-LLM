#!/bin/bash
source "$(dirname "$0")/env.sh"


python -m download -n 8

python -m download -n 400 & DATASET_DOWNLOAD_PID=$!

python -m trainer.train_tokenizer

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID