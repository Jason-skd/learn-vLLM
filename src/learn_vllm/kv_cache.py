import torch
from torch import Tensor


class KVCache:
    """Stores key/value tensors for one unpadded request.

    Each decoder layer has an independent dense buffer with shape
    [B, Hkv, Lmax, D]. `length` is shared by all
    layers because one forward pass writes the same tokens into every layer.

    Shape notation:
        B: Batch size, which is always 1 for this cache.
        Hkv: Number of key/value attention heads.
        Lmax: Maximum number of cached tokens.
        T: Number of new tokens processed by the current forward pass.
            T equals the prompt length during prefill and is usually 1 during
            decode.
        Lused: Number of valid cached tokens after a write. Lused = length + T.
        D: Dimension of one attention head.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        # ---------- 1. validate dimensions ----------
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_kv_heads <= 0 or head_dim <= 0:
            raise ValueError("num_kv_heads and head_dim must be positive.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        # ---------- 2. store cache configuration and shared length ----------
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = torch.device(device)

        # length is a scalar shared by every decoder layer:
        # it is the number of valid tokens.
        self.length = 0

        # ---------- 3. allocate one dense buffer per decoder layer ----------
        shape = (1, num_kv_heads, max_seq_len, head_dim)
        # shape: [B=1, Hkv, Lmax, D]
        # Each layer has different projection weights, so keys and values
        # cannot be shared across decoder layers.
        self._keys = [
            torch.empty(shape, dtype=dtype, device=self.device)
            for _ in range(num_layers)
        ]
        # _keys[layer_idx]: [B, Hkv, Lmax, D]
        self._values = [
            torch.empty(shape, dtype=dtype, device=self.device)
            for _ in range(num_layers)
        ]
        # _values[layer_idx]: [B, Hkv, Lmax, D]

    def ensure_capacity(self, token_count: int) -> None:
        """Checks that the cache can hold the upcoming tokens.

        This method does not change the cache length. It only validates the
        requested write range before any decoder layer modifies the buffers.
        """
        # ---------- 1. validate token count ----------
        if token_count <= 0:
            raise ValueError("token_count must be positive.")

        # ---------- 2. check the end position ----------
        end = self.length + token_count
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache capacity exceeded: need {end}, "
                f"capacity is {self.max_seq_len}."
            )

    def append(
        self,
        layer_idx: int,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Writes new keys and values for one decoder layer.

        Args:
            layer_idx: Index of the decoder layer being updated.
            key: [B, Hkv, T, D]
            value: [B, Hkv, T, D]
        Returns:
            keys: [B, Hkv, Lused, D]
            values: [B, Hkv, Lused, D]
        """
        # ---------- 1. validate layer and tensor shapes ----------
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError("layer_idx is out of range.")
        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape.")
        if key.ndim != 4:
            raise ValueError("key and value must have shape [B, Hkv, T, D].")
        if key.shape[0] != 1 or key.shape[1] != self.num_kv_heads:
            raise ValueError(
                "key and value have incompatible batch or head dimensions."
            )
        if key.shape[-1] != self.head_dim:
            raise ValueError("key and value have an incompatible head dimension.")
        if key.dtype != self.dtype or key.device != self.device:
            raise ValueError("key and value must match the cache dtype and device.")

        token_count = key.shape[-2]
        self.ensure_capacity(token_count)

        # ---------- 2. calculate the shared write range ----------
        start = self.length
        end = start + token_count
        # start/end are scalar positions that are the same for every layer.

        # ---------- 3. write this layer's K/V into the valid segment ----------
        self._keys[layer_idx][:, :, start:end, :].copy_(key)
        self._values[layer_idx][:, :, start:end, :].copy_(value)
        # Written slices: [B=1, Hkv, T, D]

        # ---------- 4. return the complete valid prefix for attention ----------
        keys = self._keys[layer_idx][:, :, :end, :]
        values = self._values[layer_idx][:, :, :end, :]
        # keys/values: [B=1, Hkv, Lused, D]
        return keys, values

    def advance(self, token_count: int) -> None:
        """Marks tokens written by all decoder layers as valid.

        The model calls this method once, after every layer has appended the same
        token range. Advancing inside append() would make later layers start from
        the wrong position.
        """
        # ---------- 1. validate the shared cache range ----------
        self.ensure_capacity(token_count)

        # ---------- 2. advance the shared valid-prefix length ----------
        self.length += token_count
