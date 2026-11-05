import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import Qwen3Config

from .kv_cache import KVCache


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


def causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    query_positions: Tensor,
    *,
    key_mask: Tensor | None = None,
    query_mask: Tensor | None = None,
) -> Tensor:
    """Computes grouped-query causal attention with optional padding masks.

    Args:
        q: [B, Hq, T, D]
        k: [B, Hkv, Lused, D]
        v: [B, Hkv, Lused, D]
        query_positions: [B, T], absolute positions of the new queries.
        key_mask: [B, Lused], True for valid key/value positions.
        query_mask: [B, T], True for valid query positions.
    Returns:
        [B, Hq, T, D]. Queries without visible keys return zeros.

    Key columns represent consecutive positions starting at zero.
    This supports right padding and compact continuous cache prefixes.
    """
    # ---------- 1. validate tensor shapes and attention heads ----------
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Queries, keys and values must be rank-4 tensors.")

    query_heads, kv_heads = q.shape[1], k.shape[1]
    if kv_heads <= 0 or query_heads <= 0 or query_heads % kv_heads != 0:
        raise ValueError("Query heads must be a positive multiple of KV heads.")
    if k.shape != v.shape:
        raise ValueError("Keys and values must have the same shape.")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Queries, keys and values must have matching B and D.")

    batch, _, token_count, head_dim = q.shape
    key_length = k.shape[-2]
    if key_length == 0:
        raise ValueError("Keys and values must contain at least one position.")
    if query_positions.shape != (batch, token_count):
        raise ValueError("query_positions must have shape [B, T].")

    # ---------- 2. prepare valid key and query masks ----------
    if key_mask is None:
        key_mask = torch.ones((batch, key_length), dtype=torch.bool, device=q.device)
    if query_mask is None:
        query_mask = torch.ones((batch, token_count), dtype=torch.bool, device=q.device)

    for name, mask, shape in (
        ("key_mask", key_mask, (batch, key_length)),
        ("query_mask", query_mask, (batch, token_count)),
    ):
        if mask.shape != shape:
            raise ValueError(f"{name} must have shape {shape}.")
        if mask.dtype != torch.bool or mask.device != q.device:
            raise ValueError(
                f"{name} must be boolean and on the same device as queries."
            )

    # ---------- 3. repeat KV heads to match query heads ----------
    groups = query_heads // kv_heads
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    # k/v: [B, Hq, Lused, D]

    # ---------- 4. compute attention scores ----------
    scores = (q @ k.transpose(-2, -1)) * (head_dim**-0.5)
    # scores: [B, Hq, T, Lused]

    # ---------- 5. combine causal and padding constraints ----------
    key_positions = torch.arange(key_length, device=q.device)
    causal = key_positions.view(1, 1, -1) <= query_positions.unsqueeze(-1)
    # causal: [B, T, Lused]

    allowed = causal & key_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
    # allowed: [B, T, Lused]
    # A query can attend only to valid keys at or before its own position.

    allowed = allowed.unsqueeze(1)
    # allowed: [B, 1, T, Lused], shared by all query heads

    scores = scores.masked_fill(~allowed, float("-inf"))

    # ---------- 6. handle queries without any visible key ----------
    has_keys = allowed.any(dim=-1, keepdim=True)
    # has_keys: [B, 1, T, 1]

    # Softmax over only -inf values produces NaN.
    # Use finite scores for these rows, then clear their probabilities below.
    scores = scores.masked_fill(~has_keys, 0.0)

    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    probabilities = probabilities.masked_fill(~allowed, 0.0)
    # probabilities: [B, Hq, T, Lused]
    # Invalid query rows now contain only zeros.

    # ---------- 7. combine values using attention probabilities ----------
    return probabilities @ v
    # output: [B, Hq, T, D]


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

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        position_ids: Tensor,
        *,
        cache: KVCache | None = None,
        layer_idx: int | None = None,
        key_mask: Tensor | None = None,
        query_mask: Tensor | None = None,
    ) -> Tensor:
        """Projects, normalizes and rotates queries and keys before attention.

        Args:
            x: [B, T, hidden_size]
            cos: [B, T, D]
            sin: [B, T, D]
            position_ids: [B, T] absolute token positions.
            cache: Optional KV cache for this request.
            layer_idx: Index of this attention layer in the decoder.
            key_mask: [B, Lused], True for valid key/value positions.
            query_mask: [B, T], True for valid new query positions.
        Returns:
            [B, T, hidden_size]
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
        # q: [B, Hq, T, D]; new k/v: [B, Hkv, T, D]
        q, k = apply_rope(q, k, cos, sin)

        # ---------- 3. append valid new K/V and read the cache prefix ----------
        if cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx is required when cache is provided.")
            k, v = cache.append(
                layer_idx,
                k,
                v,
                token_mask=query_mask,
            )
            # k, v: [B, Hkv, Lused, D]

        # ---------- 4. compute attention and merge heads ----------
        output = causal_attention(
            q,
            k,
            v,
            position_ids,
            key_mask=key_mask,
            query_mask=query_mask,
        )
        # output: [B, Hq, T, D]

        output = output.transpose(1, 2).reshape(batch, length, -1)
        # output: [B, T, Hq * D], which need not equal hidden_size
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    """Decoder Layer: Combines attention and MLP with pre-normalized residuals."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        position_ids: Tensor,
        *,
        cache: KVCache | None = None,
        layer_idx: int | None = None,
        key_mask: Tensor | None = None,
        query_mask: Tensor | None = None,
    ) -> Tensor:
        """Applies attention and the feed-forward transformation.

        Args:
            x: [B, T, hidden_size]
            cos: [B, T, D]
            sin: [B, T, D]
            position_ids: [B, T] absolute token positions.
            cache: Optional KV cache for this request.
            layer_idx: Index of this attention layer in the decoder.
            key_mask: [B, Lused], True for valid key/value positions.
            query_mask: [B, T], True for valid new query positions.
        Returns:
            [B, T, hidden_size]
        """
        # ---------- 1. attention with residual connection ----------
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(
            x,
            cos,
            sin,
            position_ids,
            cache=cache,
            layer_idx=layer_idx,
            key_mask=key_mask,
            query_mask=query_mask,
        )
        x = residual + x

        # ---------- 2. MLP with residual connection ----------
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class Qwen3Model(nn.Module):
    """Qwen3 Backbone: Runs right-padded batches with optional continuous cache.

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

    def forward(
        self,
        input_ids: Tensor,
        *,
        token_mask: Tensor | None = None,
        cache: KVCache | None = None,
    ) -> Tensor:
        """Embeds tokens and runs all decoder layers.

        Args:
            input_ids: [B, T], token IDs with optional right padding.
            token_mask: [B, T], True for real query tokens. Empty rows are
                allowed when other requests remain active.
            cache: Optional cache with the same fixed batch size as input_ids.
        Returns:
            [B, T, hidden_size]. Padding positions must be ignored.
        """
        # ---------- 1. validate input dimensions ----------
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [B, T] with B, T > 0.")

        batch, token_count = input_ids.shape

        # ---------- 2. prepare and validate the right-padding mask ----------
        if token_mask is None:
            token_mask = torch.ones(
                input_ids.shape,
                dtype=torch.bool,
                device=input_ids.device,
            )

        if token_mask.shape != input_ids.shape:
            raise ValueError("token_mask must have the same shape as input_ids.")
        if token_mask.dtype != torch.bool or token_mask.device != input_ids.device:
            raise ValueError(
                "token_mask must be boolean and on the same device as input_ids."
            )
        # A right-padded row may change from True to False, never back to True.
        padding_then_token = ~token_mask[:, :-1] & token_mask[:, 1:]
        if padding_then_token.any().item():
            raise ValueError("token_mask must describe right padding.")

        if not token_mask.any().item():
            raise ValueError("token_mask must contain an active token.")

        # ---------- 3. prepare cache position and visibility metadata ----------
        if cache is not None:
            if cache.batch_size != batch:
                raise ValueError("cache batch size must match input_ids.")

            # Requests resume at different absolute positions.
            starts = cache.lengths.unsqueeze(1)
            positions = starts + torch.arange(token_count, device=input_ids.device)
            key_mask = cache.projected_key_mask(token_mask)
        else:
            # Without a cache, keys and queries come from the same input.
            positions = (
                torch.arange(token_count, device=input_ids.device)
                .unsqueeze(0)
                .expand(batch, -1)
            )
            key_mask = token_mask

        # positions: [B, T]
        # key_mask: [B, Lused] with cache, or [B, T] without a cache.

        # ---------- 4. embed tokens and compute rotary embeddings ----------
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(positions, x.dtype)

        # ---------- 5. run decoder layers with the same visibility masks ----------
        for layer_idx, layer in enumerate(self.layers):
            x = layer(
                x,
                cos,
                sin,
                positions,
                key_mask=key_mask,
                query_mask=token_mask,
                cache=cache,
                layer_idx=layer_idx,
            )

        # ---------- 6. advance each request after every layer has written ----------
        if cache is not None:
            cache.advance(token_mask)

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

    def create_kv_cache(
        self,
        max_seq_len: int,
        *,
        batch_size: int,
    ) -> KVCache:
        """Creates a cache configured from this model's attention layers.

        Args:
            max_seq_len: Maximum number of prompt and generated tokens.
            batch_size: Fixed number of requests occupying batch slots.
        Returns:
            An empty continuous KV cache.
        """
        # ---------- 1. read dimensions from the loaded model ----------
        embedding = self.model.embed_tokens.weight
        attention = self.model.layers[0].self_attn

        # ---------- 2. allocate cache with the model's runtime dtype and device ----------
        return KVCache(
            num_layers=len(self.model.layers),
            num_kv_heads=attention.num_kv_heads,
            head_dim=attention.head_dim,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            dtype=embedding.dtype,
            device=embedding.device,
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        token_mask: Tensor | None = None,
        cache: KVCache | None = None,
    ) -> Tensor:
        """Computes logits for every input position.

        Args:
            input_ids: [B, T], token IDs with optional right padding.
            token_mask: [B, T], True for real tokens.
            cache: Optional cache with the same fixed batch size as input_ids.
        Returns:
            [B, T, vocab_size]. Padding positions must be ignored.
        """
        hidden_states = self.model(
            input_ids,
            token_mask=token_mask,
            cache=cache,
        )
        return self.lm_head(hidden_states)
