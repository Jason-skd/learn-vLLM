import torch
from torch import Tensor


class KVCache:
    """Stores continuous key/value tensors for a fixed batch of requests.

    Every request occupies one row of a dense cache. `lengths[b]` records
    how many token positions are valid for request b. All decoder layers
    share the same lengths because one forward pass writes the same positions
    into every layer.

    Shape notation:
        B: Fixed number of batch slots.
        Hkv: Number of key/value attention heads.
        Lmax: Maximum number of cached tokens per request.
        T: Number of new positions in the current forward pass.
        Lused: Longest valid cache prefix in the current batch.
        D: Dimension of one attention head.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        # ---------- 1. validate cache dimensions ----------
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_kv_heads <= 0 or head_dim <= 0:
            raise ValueError("num_kv_heads and head_dim must be positive.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        # ---------- 2. store configuration and request lengths ----------
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = torch.device(device)

        # lengths: [B], one independent valid-prefix length per request.
        self.lengths = torch.zeros(
            batch_size,
            dtype=torch.int64,
            device=self.device,
        )

        # ---------- 3. allocate one continuous buffer per decoder layer ----------
        shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        # shape: [B, Hkv, Lmax, D]

        # Zero initialization is safer than uninitialized memory because a
        # shorter request may share a padded attention range with a longer one.
        # Its unwritten positions are also excluded by the key mask.
        self._keys = [
            torch.zeros(shape, dtype=dtype, device=self.device)
            for _ in range(num_layers)
        ]
        self._values = [
            torch.zeros(shape, dtype=dtype, device=self.device)
            for _ in range(num_layers)
        ]
        # _keys/layer_idx: [B, Hkv, Lmax, D]
        # _values/layer_idx: [B, Hkv, Lmax, D]

    def _next_lengths(
        self,
        token_mask: Tensor,
        *,
        token_count: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Calculates token counts and lengths after this forward pass."""
        # ---------- 1. validate the mask and its dimensions ----------
        if token_mask.ndim != 2 or token_mask.shape[0] != self.batch_size:
            raise ValueError("token_mask must have shape [batch_size, T].")
        if token_count is not None and token_mask.shape[1] != token_count:
            raise ValueError("token_mask must match the new token count.")
        if token_mask.dtype != torch.bool or token_mask.device != self.device:
            raise ValueError(
                "token_mask must be boolean and on the same device as the cache."
            )

        # Right padding means each row is a valid prefix followed by padding.
        padding_then_token = ~token_mask[:, :-1] & token_mask[:, 1:]
        if padding_then_token.any().item():
            raise ValueError("token_mask must describe right padding.")

        # ---------- 2. calculate independent request lengths ----------
        counts = token_mask.sum(dim=1)
        # counts: [B], number of new valid tokens per request.

        next_lengths = self.lengths + counts
        # next_lengths: [B], lengths after this forward pass.

        required = int(next_lengths.max().item())
        if required > self.max_seq_len:
            raise ValueError(
                f"KV cache capacity exceeded: need {required}, "
                f"capacity is {self.max_seq_len}."
            )

        return counts, next_lengths

    def projected_key_mask(self, token_mask: Tensor) -> Tensor:
        """Returns valid cached keys after this forward pass.

        This method is called before the first decoder layer appends K/V.
        It projects what each request length will be after all layers finish.

        Args:
            token_mask: [B, T], True for new tokens that should be cached.
        Returns:
            [B, Lused], True for valid cache positions in this forward pass.
        """
        # ---------- 1. calculate lengths after the current write ----------
        _, next_lengths = self._next_lengths(token_mask)
        used = int(next_lengths.max().item())

        # ---------- 2. mark the valid compact prefix of every request ----------
        positions = torch.arange(used, device=self.device)
        mask = positions.unsqueeze(0) < next_lengths.unsqueeze(1)

        # Example with lengths [3, 1] and used=3:
        # [[True, True, True],
        #  [True, False, False]]
        return mask

    def append(
        self,
        layer_idx: int,
        key: Tensor,
        value: Tensor,
        *,
        token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Writes new keys and values for one decoder layer.

        Args:
            layer_idx: Index of the decoder layer being updated.
            key: [B, Hkv, T, D]
            value: [B, Hkv, T, D]
            token_mask: [B, T], True for new tokens that should be written.
        Returns:
            keys: [B, Hkv, Lused, D]
            values: [B, Hkv, Lused, D]
        """
        # ---------- 1. validate the layer and tensor shapes ----------
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError("layer_idx is out of range.")
        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape.")
        if key.ndim != 4:
            raise ValueError("key and value must have shape [B, Hkv, T, D].")
        if key.shape[0] != self.batch_size or key.shape[1] != self.num_kv_heads:
            raise ValueError(
                "key and value have incompatible batch or head dimensions."
            )
        if key.shape[-2] == 0:
            raise ValueError("key and value must contain new tokens.")
        if key.shape[-1] != self.head_dim:
            raise ValueError("key and value have an incompatible head dimension.")
        if key.dtype != self.dtype or key.device != self.device:
            raise ValueError("key and value must match the cache dtype and device.")

        # ---------- 2. calculate each request's write range ----------
        counts, next_lengths = self._next_lengths(
            token_mask,
            token_count=key.shape[-2],
        )
        used = int(next_lengths.max().item())

        # ---------- 3. write each request's valid K/V prefix ----------
        for batch_idx in range(self.batch_size):
            count = int(counts[batch_idx].item())
            if count == 0:
                # A finished request keeps its cache but adds no new K/V.
                continue

            start = int(self.lengths[batch_idx].item())
            end = start + count

            self._keys[layer_idx][batch_idx : batch_idx + 1, :, start:end, :].copy_(
                key[batch_idx : batch_idx + 1, :, :count, :]
            )
            self._values[layer_idx][batch_idx : batch_idx + 1, :, start:end, :].copy_(
                value[batch_idx : batch_idx + 1, :, :count, :]
            )
            # New K/V slice: [1, Hkv, count, D]

        # ---------- 4. return the dense prefix used by attention ----------
        keys = self._keys[layer_idx][:, :, :used, :]
        values = self._values[layer_idx][:, :, :used, :]
        # keys/values: [B, Hkv, Lused, D]

        # Shorter rows still contain unused columns inside the dense prefix.
        # projected_key_mask() excludes those columns from attention.
        return keys, values

    def advance(self, token_mask: Tensor) -> None:
        """Marks new tokens as valid after every decoder layer has written."""
        # append() reads lengths without modifying them, so every layer writes
        # into the same start positions. Only this call advances the metadata.
        _, next_lengths = self._next_lengths(token_mask)
        self.lengths.copy_(next_lengths)
