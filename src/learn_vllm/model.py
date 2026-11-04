import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import Qwen3Config


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization: Normalizes the last dimension and
    then multiplies it by a learnable weight.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalizes the last dimension and then multiplies it by a learnable weight.

        Args:
            x: [..., dim]
        Returns:
            [..., dim]
        """
        values = x.float()
        mean_square = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(mean_square + self.eps)
        return normalized.to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Embedding: Computes cosine and sine values for default RoPE."""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer.")

        exponents = torch.arange(0, head_dim, 2) / head_dim
        # Match the reference operation order to avoid float32 rounding drift.
        inv_freq = 1.0 / (theta**exponents)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, position_ids: Tensor, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        """Computes the rotary embeddings for the given position IDs.
        D: Dimension, here it refers to head_dim (the dimension of a single
        attention head)

        Args:
            position_ids: [B, L]
            dtype: The data type to convert the output to.
        Returns:
            cos_angles: [B, L, D]
            sin_angles: [B, L, D]
        """
        angles = position_ids.float().unsqueeze(-1) * self.inv_freq
        angles = torch.cat((angles, angles), dim=-1)
        # angles: [B, L, D]

        return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Rotates pairs formed by the first and second halves of the last dimension.

    Args:
        x: [..., D], where D is even.
    Returns:
        [..., D]
    """
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Applies rotary positional embeddings to queries and keys.

    Args:
        q: [B, Hq, L, D]
        k: [B, Hkv, L, D]
        cos: [B, L, D]
        sin: [B, L, D]
    Returns:
        q: [B, Hq, L, D]
        k: [B, Hkv, L, D]
    """
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    # cos/sin: [B, 1, L, D], broadcast across attention heads

    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def causal_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Computes grouped-query causal attention without padding or KV cache.

    Args:
        q: [B, Hq, L, D]
        k: [B, Hkv, L, D]
        v: [B, Hkv, L, D]
    Returns:
        [B, Hq, L, D]
    """
    # ---------- 1. obtain the quantity of attention heads ----------
    query_heads, kv_heads = q.shape[1], k.shape[1]
    if kv_heads <= 0 or query_heads <= 0 or query_heads % kv_heads != 0:
        raise ValueError("Query heads must be a positive multiple of KV heads.")
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("Queries, keys and values must have matching B, L and D.")

    # --------- 2. duplicate the KV header ----------
    groups = query_heads // kv_heads
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    # k/v: [B, Hq, L, D], notice that Hkv -> Hq

    # --------- 3. compute the attention ----------
    length, head_dim = q.shape[-2:]
    scores = (q @ k.transpose(-2, -1)) * (head_dim**-0.5)
    # scores: [B, Hq, L, L]

    # --------- 4. mask out future tokens ----------
    future = torch.ones(length, length, dtype=torch.bool, device=q.device).triu(1)
    scores = scores.masked_fill(future, float("-inf"))
    # Rows are query positions; columns beyond each query are future tokens.

    # --------- 5. calculate the probability ----------
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return probabilities @ v


class MLP(nn.Module):
    """Gated MLP: Combines a SiLU gate with an up projection, then projects down."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Applies the gated feed-forward transformation.

        Args:
            x: [B, L, hidden_size]
        Returns:
            [B, L, hidden_size]
        """
        gate = F.silu(self.gate_proj(x))
        values = self.up_proj(x)
        # gate/values: [B, L, intermediate_size]

        return self.down_proj(gate * values)


class Attention(nn.Module):
    """Self Attention: Projects hidden states and applies grouped-query attention."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Projects, normalizes and rotates queries and keys before attention.

        Args:
            x: [B, L, hidden_size]
            cos: [B, L, D]
            sin: [B, L, D]
        Returns:
            [B, L, hidden_size]
        """
        # ---------- 1. project and split attention heads ----------
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)

        # ---------- 2. normalize queries and keys, then apply RoPE ----------
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        # q: [B, Hq, L, D]; k/v: [B, Hkv, L, D]
        q, k = apply_rope(q, k, cos, sin)

        # ---------- 3. compute attention and merge heads ----------
        output = causal_attention(q, k, v)
        output = output.transpose(1, 2).reshape(batch, length, -1)
        # output: [B, L, Hq * D], which need not equal hidden_size
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    """Decoder Layer: Combines attention and MLP with pre-normalized residuals."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Applies attention and the feed-forward transformation.

        Args:
            x: [B, L, hidden_size]
            cos: [B, L, D]
            sin: [B, L, D]
        Returns:
            [B, L, hidden_size]
        """
        # ---------- 1. attention with residual connection ----------
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin)
        x = residual + x

        # ---------- 2. MLP with residual connection ----------
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class Qwen3Model(nn.Module):
    """Qwen3 Backbone: Runs one unpadded sequence without a KV cache.

    Only default RoPE, full attention and SiLU are supported. Dropout must be
    zero; this implementation is intended for inference with loaded weights.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        # ---------- 1. check the supported model configuration ----------
        if config.hidden_act != "silu":
            raise ValueError("Only SiLU activation is supported.")
        if config.attention_dropout != 0:
            raise ValueError("Attention dropout must be zero.")
        if any(layer_type != "full_attention" for layer_type in config.layer_types):
            raise ValueError("Only full attention is supported.")
        rope = config.rope_parameters
        if rope.get("rope_type", "default") != "default":
            raise ValueError("Only default RoPE is supported.")
        if (
            config.num_attention_heads <= 0
            or config.num_key_value_heads <= 0
            or config.num_attention_heads % config.num_key_value_heads != 0
        ):
            raise ValueError("Query heads must be a positive multiple of KV heads.")

        # ---------- 2. build embeddings, decoder layers and final norm ----------
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.rotary_emb = RotaryEmbedding(config.head_dim, rope["rope_theta"])
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Embeds input tokens and runs all decoder layers.

        Args:
            input_ids: [1, L], a nonempty sequence without padding.
        Returns:
            [1, L, hidden_size]
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [1, L] with L > 0.")

        # ---------- 1. embed tokens and compute shared position embeddings ----------
        x = self.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(
            0
        )
        cos, sin = self.rotary_emb(positions, x.dtype)
        # cos/sin: [1, L, D], reused by every decoder layer

        # ---------- 2. run decoder layers and normalize ----------
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 Causal LM: Maps the final hidden states to vocabulary logits."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            # Share the parameter itself so loading embeddings also updates the head.
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        """Computes logits for every input position without generating tokens.

        Args:
            input_ids: [1, L], a nonempty sequence without padding.
        Returns:
            [1, L, vocab_size]
        """
        hidden_states = self.model(input_ids)
        return self.lm_head(hidden_states)
