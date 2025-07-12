import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        scores = scores + mask
    return mx.matmul(softmax(scores, axis=-1), value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        batch_size = query.shape[0]
        # query, key, value: N x L x E

        # 0. Compute linear projections
        query = linear(query, self.wq)
        key = linear(key, self.wk)
        value = linear(value, self.wv)

        # 1. Reshape to (batch_size, num_heads, seq_len, hidden_size)
        query = mx.reshape(
            query, (batch_size, -1, self.num_heads, self.hidden_size // self.num_heads))
        key = mx.reshape(
            key, (batch_size, -1, self.num_heads, self.hidden_size // self.num_heads))
        value = mx.reshape(
            value, (batch_size, -1, self.num_heads, self.hidden_size // self.num_heads))

        # 2. Transpose to (batch_size, num_heads, seq_len, hidden_size)
        query = mx.transpose(query, axes=(0, 2, 1, 3))
        key = mx.transpose(key, axes=(0, 2, 1, 3))
        value = mx.transpose(value, axes=(0, 2, 1, 3))

        # 3. Compute scaled dot product attention
        output = scaled_dot_product_attention_simple(
            query, key, value, mask=mask)

        # 4. concat heads
        output = mx.reshape(mx.transpose(output, axes=(
            0, 2, 1, 3)), (batch_size, -1, self.hidden_size))

        # 5. Compute linear projection
        output = linear(output, self.wo)

        return output


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    # 实现一个 causal mask，返回 shape 为 (L, S) 的下三角矩阵，不可见位置为 -inf，可见为 0
    # 其中第 i 行的前 S-L+i 个元素为 -inf，其余为 0（即对齐到右下角）
    # 例如 L=3, S=5:
    # [[ 0,  0,  0, -inf, -inf],
    #  [ 0,  0,  0,  0,  -inf],
    #  [ 0,  0,  0,  0,   0 ]]
    # 正确实现：用tril生成右下对齐的下三角
    # 第1行：生成一个 shape 为 (L, S) 的布尔下三角矩阵，右下对齐
    # mx.ones((L, S), dtype=mx.bool_) 生成全为 True 的 (L, S) 矩阵
    # mx.tril(..., k=(S-L)) 取下三角，k=(S-L) 保证下三角区域右下对齐
    mask = mx.tril(mx.ones((L, S), dtype=mx.bool_), k=(S - L))
    # 第2行：将 mask 中为 True 的位置赋值为 0（可见），为 False 的位置赋值为 -inf（不可见）
    # mx.where(mask, mx.array(0, dtype=dtype), mx.array(-mx.inf, dtype=dtype))
    # 这样最终得到的 mask，右下三角为 0，其余为 -inf
    mask = mx.where(mask, mx.array(0, dtype=dtype),
                    mx.array(-mx.inf, dtype=dtype))
    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    """
    实现 grouped attention（分组注意力），即将 head 分为 group，每组内部做 attention，组间不做 attention。
    假设 query, key, value 的 shape: (batch, num_heads, seq_len, head_dim)
    其中 num_heads = num_groups * heads_per_group
    mask: 可选，shape 可为 (batch, 1, seq_len, seq_len) 或 (batch, num_heads, seq_len, seq_len)
    """
    factor = mx.rsqrt(query.shape[-1]) if scale is None else mx.array(scale)
    factor = factor.astype(query.dtype)
    expected_shape = query.shape

    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]
    assert H_q % H == 0
    n_repeats = H_q // H

    query = query.reshape(-1, H, n_repeats, L, D)
    key = key.reshape(-1, H, 1, S, D)
    value = value.reshape(-1, H, 1, S, D)

    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        if mask == "causal":
            mask = causal_mask(L, S, scores.dtype)
            scores = scores + mask
        else:
            mask = mask.reshape(-1, H, n_repeats,
                                mask.shape[-2], mask.shape[-1])
            scores = scores + mask
    result = mx.matmul(softmax(scores, axis=-1), value)
    return result.reshape(expected_shape)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
