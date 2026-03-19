from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from chat_llm.model.attention import flash_attn
from chat_llm.utils.common import COMPUTE_DTYPE, print_master


@dataclass
class ModelConfig:
    max_seq_len: int = 2048
    vocab_size: int = 32768
    n_layers: int = 12
    n_q_heads: int = 6
    n_kv_heads: int = 6
    embed_dim: int = 768
    d_ff: int = 2048  #  floor(embed_dim * 8/3 / 64) * 64

    # Sliding window pattern:
    # L: Full context attention
    # S: Sliding window attention
    # SSSL: 3 layers of sliding window attention followed by 1 layer of full context attention
    window_pattern: str = "SSSL"  #

    def __post_init__(self):
        assert self.embed_dim % self.n_q_heads == 0, "embed_dim must be divisible by n_q_heads"
        window_pattern_len = len(self.window_pattern)
        assert window_pattern_len > 0, "window_pattern must not be empty"
        assert all(c in "LS" for c in self.window_pattern), "window_pattern must only contain 'L' and 'S'"
        assert self.n_layers % window_pattern_len == 0, (
            "n_layers must be a multiple of the length of window_pattern"
        )
        assert self.n_q_heads % self.n_kv_heads == 0, "n_q_heads must be divisible by n_kv_heads"


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    "No learnable parameters, just normalizes the input."
    D = x.shape[-1]
    return F.rms_norm(x, (D,))


class Linear(nn.Linear):
    """
    A linear layer that supports mixed precision by converting weights and bias to the input dtype during the forward pass.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        weight = self.weight.to(dtype)
        bias = self.bias.to(dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


def apply_rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x = cos * x
    """
    assert x.ndim == 4, "Input tensor must be of shape (batch_size, seq_len, num_heads, head_dim)"
    assert cos.shape == sin.shape, "Cosine and sine tensors must have the same shape"
    assert cos.ndim == 4 and sin.ndim == 4, (
        "Cosine and sine tensors must be of shape (1, seq_len, num_heads, head_dim // 2)"
    )

    x1, x2 = x.chunk(2, dim=-1)

    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def has_ve(layer_idx: int, n_layers: int) -> bool:
    # VE in odd layers if n_layers is even, else VE in even layers
    return layer_idx % 2 == (n_layers - 1) % 2


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        """
        layer_idx:
            1. Determined wether use full context attention of sliding window attention
            2. Used for KV caching in the future
            3. Determined wether use value embeddings
        """
        super().__init__()

        self.layer_idx = layer_idx
        self.n_q_heads = config.n_q_heads
        self.n_kv_heads = config.n_kv_heads

        assert config.embed_dim % self.n_q_heads == 0, "embed_dim must be divisible by n_q_heads"
        self.head_dim = config.embed_dim // self.n_q_heads
        assert self.n_q_heads % self.n_kv_heads == 0, "n_q_heads must be divisible by n_kv_heads"

        self.q_proj = Linear(config.embed_dim, config.n_q_heads * self.head_dim, bias=False)
        self.k_proj = Linear(config.embed_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(config.embed_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = Linear(config.n_q_heads * self.head_dim, config.embed_dim, bias=False)

        # Value Embeddings
        self.ve_gate_channels = 12
        self.ve_gate = (
            Linear(self.ve_gate_channels, self.n_kv_heads, bias=False)
            if has_ve(layer_idx, config.n_layers)
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        ve: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        window_size: int,
        kv_cache,
    ) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        if ve is not None and self.ve_gate is not None:
            ve = ve.view(B, T, self.n_kv_heads, self.head_dim)
            gate = 3 * F.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))  # (B, n_kv_heads
            v = v + gate.unsqueeze(-1) * ve

        # Apply rotary embeddings to q and k
        q = apply_rotary_embedding(q, cos, sin)
        k = apply_rotary_embedding(k, cos, sin)

        # Apply QK-Norm
        q = rms_norm(q)
        k = rms_norm(k)

        if kv_cache is None:
            out = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference with KV caching
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            out = flash_attn.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k,
                v,
                cache_seqlens=kv_cache.cache_seq_lens,
                causal=True,
                window_size=window_size,
            )

            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        out = out.contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.up_proj = Linear(config.embed_dim, config.d_ff, bias=False)
        self.down_proj = Linear(config.d_ff, config.embed_dim, bias=False)
        self.gate_proj = Linear(config.embed_dim, config.d_ff, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        return self.down_proj(F.silu(gate) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos, sin, window_size, kv_cache):
        x = x + self.attn(rms_norm(x), ve, cos, sin, window_size, kv_cache)
        x = x + self.mlp(rms_norm(x))
        return x


def pre_compute_cos_sin(
    max_seq_len: int,
    head_dim: int,
    base: int = 10_000,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    channel_range = torch.arange(head_dim // 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (channel_range / (head_dim // 2)))
    pos_ids = torch.arange(max_seq_len, device=device, dtype=torch.float32)

    freqs = torch.einsum("i,j->ij", pos_ids, inv_freq)  # (max_seq_len, head_dim // 2)
    cos = freqs.cos()[None, :, None, :]  # (1, max_seq_len, 1, head_dim // 2)
    sin = freqs.sin()[None, :, None, :]  # (1, max_seq_len, 1, head_dim // 2)
    return cos.to(dtype), sin.to(dtype)


class LLMModel(nn.Module):
    def __init__(self, config: ModelConfig, padded_vocab_size: int = 64):
        super().__init__()

        self.config = config
        self.window_sizes = self._compute_window_size(config)

        padded_vocab_size = (
            (config.vocab_size + padded_vocab_size - 1) // padded_vocab_size
        ) * padded_vocab_size
        if padded_vocab_size != config.vocab_size:
            print_master(
                f"Warning: vocab_size {config.vocab_size} is not divisible by padded_vocab_size {padded_vocab_size}, "
                f"padding to {padded_vocab_size}"
            )

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.embed_dim),
                "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layers)]),
            }
        )

        self.lm_head = Linear(config.embed_dim, padded_vocab_size, bias=False)

        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layers))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layers))

        head_dim = config.embed_dim // config.n_q_heads
        kv_dim = config.n_kv_heads * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(padded_vocab_size, kv_dim)
                for i in range(config.n_layers)
                if has_ve(i, config.n_layers)
            }
        )

        self.rotary_seq_len = config.max_seq_len * 10
        cos, sin = pre_compute_cos_sin(self.rotary_seq_len, head_dim, device=self.device, dtype=COMPUTE_DTYPE)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _compute_window_size(self, config: ModelConfig) -> list[tuple[int, int]]:
        pattern = config.window_pattern.upper()

        long_window = config.max_seq_len
        short_window = (config.max_seq_len // 3 // 128) * 128
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}

        window_sizes = []
        for layer_idx in range(config.n_layers):
            pattern_idx = layer_idx % len(pattern)
            window_sizes.append(char_to_window[pattern[pattern_idx]])

        window_sizes[-1] = (long_window, 0)  # Ensure the last layer always has full context attention
        return window_sizes

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def init_weights(self):
        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.08)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.embed_dim
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            nn.init.uniform_(block.attn.q_proj.weight, -s, s)
            nn.init.uniform_(block.attn.k_proj.weight, -s, s)
            nn.init.uniform_(block.attn.v_proj.weight, -s, s)
            nn.init.zeros_(block.attn.out_proj.weight)

            nn.init.uniform_(block.mlp.up_proj.weight, -s * 0.5, s * 0.5)
            nn.init.uniform_(block.mlp.gate_proj.weight, -s * 0.5, s * 0.5)
            nn.init.zeros_(block.mlp.down_proj.weight)

        self.resid_lambdas.fill_(1.0)  # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)  # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        head_dim = self.config.embed_dim // self.config.n_q_heads
        cos, sin = pre_compute_cos_sin(self.rotary_seq_len, head_dim, device=self.device, dtype=COMPUTE_DTYPE)
        self.cos, self.sin = cos, sin

        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        B, T = idx.shape

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos, sin = self.cos[:, T0 : T0 + T, :, :], self.sin[:, T0 : T0 + T, :, :]

        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)

        x = rms_norm(x)

        x0 = x

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos, sin, self.window_sizes[i], kv_cache)

        x = rms_norm(x)

        # Forward Head
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., : self.config.vocab_size]  # Remove padding tokens from output logits
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction=loss_reduction,
                ignore_index=-100,
            )
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.device
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token


if __name__ == "__main__":
    config = ModelConfig()
    model = LLMModel(config)
    model.init_weights()

    print("Model parameter count:", sum(p.numel() for p in model.parameters()))
    print_master("Model initialized successfully")


def build_model_meta(depth, aspect_ratio, head_dim, vocab_size, max_seq_len, window_pattern):
    """
    Build the model on the meta device to get accurate parameter counts without using any real memory, which is important for being able to build large models without OOM issues and to compute accurate scaling law predictions for hyperparameters based on the actual parameter count of the model that will be trained.
    """
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    config = ModelConfig(
        embed_dim=model_dim,
        n_q_heads=num_heads,
        n_kv_heads=num_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_ff=4 * model_dim,
        window_pattern=window_pattern,
    )
    with torch.device("meta"):
        model_meta = LLMModel(config)
    return model_meta
