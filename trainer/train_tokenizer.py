import argparse
import os
import time

import torch

from chat_llm.tokenizer import HuggingFaceTokenizer
from chat_llm.utils.common import get_base_dir
from chat_llm.utils.data import parquets_iter_batched

# -----------------------------------------------------------------------------

DATA_DIR = os.environ.get("DATA_DIR")
TOKENIZER_DIR = os.environ.get("TOKENIZER_DIR")


parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
parser.add_argument(
    "--max-chars", type=int, default=2_000_000_000, help="Maximum characters to train on (default: 10B)"
)
parser.add_argument(
    "--doc-cap", type=int, default=10_000, help="Maximum characters per document (default: 10,000)"
)
parser.add_argument("--vocab-size", type=int, default=32768, help="Vocabulary size (default: 32768 = 2^15)")
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator


def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0
    for batch in parquets_iter_batched(DATA_DIR, split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[: args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return


base_dir = get_base_dir()
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = HuggingFaceTokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")


# Save the tokenizer to disk
tokenizer.save(TOKENIZER_DIR)

# Save the token bytes to disk
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id]  # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0)  # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8"))  # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")
