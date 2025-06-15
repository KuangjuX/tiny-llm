import mlx.core as mx


class RoPE:

    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        """Initialize RoPE (Rotary Position Embedding).

        Args:
            dims: The dimension of the input.
            seq_len: The sequence length.
            base: The base of the exponential.
            traditional: Whether to use traditional RoPE implementation.
        """
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

    def _create_cos_sin(self, dims: int, seq_len: int, base: int) -> mx.array:
        """Create cos and sin for RoPE.

        Args:
            dims: The dimension of the input.
            seq_len: The sequence length.
            base: The base of the exponential.

        Returns:
            Complex numbers containing cos and sin values for RoPE.
        """
        freqs = 1.0 / (base ** mx.arange(0, dims, 2)
                       [: dims // 2].float() / dims)
        t = mx.arange(seq_len)

        freqs = mx.outer(t, freqs).float()
        freqs_cis = mx.polar(mx.ones_like(freqs), freqs)

        return freqs_cis

    def _compute_repo(self, cos, sin, x):
        """Compute rotary position embedding.

        Args:
            cos: Cosine values.
            sin: Sine values.
            x: Input tensor.

        Returns:
            Tensor with rotary position embedding applied.
        """
        x1 = x[..., : self.dims // 2]
        x2 = x[..., self.dims // 2: self.dims]

        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos

        if self.dims < x.shape[-1]:
            rx = mx.concatenate([rx1, rx2, x[..., self.dims:]], axis=-1)
        else:
            rx = mx.concatenate([rx1, rx2], axis=-1)

        return rx

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        """Apply RoPE to the input tensor.

        Args:
            x: Input tensor.
            offset: Optional offset for position indices.

        Returns:
            Tensor with RoPE applied.
        """
        x = mx.reshape(x, (x.shape[0], x.shape[1], x.shape[2],
                           x.shape[3] // 2, 2))

        return x
