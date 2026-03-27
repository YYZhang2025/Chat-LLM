import torch


class KVCache:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim

        self.k_cache = torch.zeros(
            (num_layers, batch_size, seq_len, num_heads, head_dim), device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            (num_layers, batch_size, seq_len, num_heads, head_dim), device=device, dtype=dtype
        )

        self.cache_seq_lens = torch.zeros(batch_size, device=device, dtype=torch.int32)  # (batch_size,)

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_seq_lens.zero_()

    def get_pos(self):
        return self.cache_seq_lens[0].item()  # Assuming all sequences in the batch have the same length

    def get_layer_cache(self, layer_idx: int):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens: int):
        self.cache_seq_lens += num_tokens

    def prefill(self, other_kv: "KVCache"):
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert (
            self.n_layers == other_kv.n_layers
            and self.n_heads == other_kv.n_heads
            and self.head_dim == other_kv.head_dim
        )
        assert self.max_seq_len >= other_kv.max_seq_len

        other_pos = other_kv.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other_kv.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other_kv.v_cache[:, :, :other_pos, :, :]
        self.cache_seq_lens.fill_(other_pos)
