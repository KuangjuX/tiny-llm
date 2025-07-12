import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: (..., dim)
        # 1. Cast to float32 for numerical stability
        x_fp32 = x.astype(mx.float32)
        # 2. Compute mean of square over last dim
        rms = mx.sqrt(mx.mean(x_fp32 * x_fp32, axis=-
                      1, keepdims=True) + self.eps)
        # 3. Normalize
        x_norm = x_fp32 / rms
        # 4. Scale (cast weight to float32 for broadcast, then cast result back to input dtype)
        out = x_norm * self.weight.astype(mx.float32)
        return out.astype(x.dtype)
