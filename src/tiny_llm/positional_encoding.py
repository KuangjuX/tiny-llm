import mlx.core as mx

# =========================
# RoPE 论文与实现详细讲解
# =========================
#
# 论文出处：
#   "RoFormer: Enhanced Transformer with Rotary Position Embedding"
#   https://arxiv.org/abs/2104.09864
#
# 论文核心思想：
#   - 传统的Transformer位置编码（如Sinusoidal）是将位置信息直接加到输入向量上。
#   - RoPE（Rotary Position Embedding）提出：将位置信息编码为复数平面上的旋转操作，
#     即将每个token的特征向量视为复数对（实部+虚部），对其进行不同频率的旋转。
#   - 这样做的好处是：自注意力的点积结果会自然地包含相对位置信息（即q和k的相对旋转角度）。
#
# 公式推导：
#   - 对于每个token的特征向量x，假设其最后一维长度为d，d为偶数。
#   - 将x拆成两半，分别作为复数的实部和虚部：x = [x1, x2]，x1,x2长度均为d/2。
#   - 对于第i个位置，定义旋转角度θ_i = pos * freq_i，其中freq_i为不同维度的频率。
#   - 旋转操作：y = x1 * cosθ - x2 * sinθ + i * (x2 * cosθ + x1 * sinθ)
#   - 这样，经过旋转后的q和k做点积时，结果会带有相对位置信息。
#
# 下面结合实现详细讲解每一步：


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        """
        初始化RoPE编码器。

        参数说明：
            dims: 输入特征的最后一维长度（必须为偶数）。
            seq_len: 最大序列长度。
            base: 频率的底数，通常为10000。
            traditional: 是否采用传统实现（即最后一维reshape为[..., 2]）。
        """
        assert dims % 2 == 0, "RoPE要求dims为偶数"
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        """
        对输入x应用RoPE旋转位置编码。

        参数说明：
            x: 输入张量，形状为 (batch, seq_len, num_heads, head_dim)
            offset: 可选，指定每个batch的起止位置（支持slice或list[slice]）

        返回：
            应用RoPE后的张量，形状同输入。
        """
        # 1. 获取输入形状
        N, S, H, D = x.shape
        assert D == self.dims, f"输入最后一维{D}与RoPE设定{self.dims}不符"
        half_dims = D // 2

        # 2. 计算每个维度的频率
        #    论文公式：freq_i = base^{-i/dims}，i=0,1,...,dims/2-1
        inner = mx.arange(0, half_dims, dtype=mx.float32) / half_dims
        freqs = mx.power(self.base, -inner)  # (half_dims,)

        # 3. 计算每个token的位置索引t
        #    支持offset（如推理时的滑动窗口）
        if offset is None:
            t = mx.arange(S)  # (S,)
        elif isinstance(offset, slice):
            assert offset.stop - offset.start == S, f"offset长度应为{S}"
            t = mx.arange(offset.start, offset.stop)  # (S,)
        elif isinstance(offset, list):
            assert len(offset) == N, f"offsets长度应为batch size {N}"
            t = mx.stack([mx.arange(o.start, o.stop)
                         for o in offset], axis=0)  # (N, S)
        else:
            raise ValueError("offset类型错误")

        # 4. 计算每个位置每个频率的旋转角度
        #    pos_freqs = t * freqs
        if isinstance(t, mx.array) and t.ndim == 2:
            # t: (N, S)
            # 当 offset 是 list[slice] 时，t 的 shape 是 (N, S)，
            # 其中 N 是 batch size，S 是每个样本的序列长度。
            # 这是因为每个 batch 可能有不同的起止位置，需要为每个 batch 单独生成一组位置索引。
            # 这样 t = [ [start0, ..., stop0-1], [start1, ..., stop1-1], ... ]，shape 为 (N, S)
            # freqs 的 shape 是 (half_dims,)
            # t[:, :, None] 扩展为 (N, S, 1)，freqs[None, None, :] 扩展为 (1, 1, half_dims)
            # 广播后 pos_freqs 的 shape 就是 (N, S, half_dims)
            pos_freqs = t[:, :, None] * freqs[None, None, :]
            cos_pos = mx.cos(pos_freqs)  # (N, S, half_dims)
            sin_pos = mx.sin(pos_freqs)  # (N, S, half_dims)
            # 扩展到(N, S, H, half_dims)以便与x广播
            cos_pos = cos_pos[:, :, None, :]
            sin_pos = sin_pos[:, :, None, :]
        else:
            # t: (S,)
            pos_freqs = mx.outer(t, freqs)  # (S, half_dims)
            cos_pos = mx.cos(pos_freqs)     # (S, half_dims)
            sin_pos = mx.sin(pos_freqs)     # (S, half_dims)
            # 扩展到(N, S, H, half_dims)
            cos_pos = cos_pos[None, :, None, :]
            sin_pos = sin_pos[None, :, None, :]

        # 5. 拆分输入x的最后一维，分成两半，分别作为复数的实部和虚部
        #    论文实现有两种方式，传统方式和现代方式
        if self.traditional:
            # 传统实现：reshape为(N, S, H, half_dims, 2)
            x_ = x.reshape(N, S, H, half_dims, 2)
            x1 = x_[..., 0]  # 实部
            x2 = x_[..., 1]  # 虚部
        else:
            # 现代实现：直接切分
            x1 = x[..., :half_dims]      # 实部
            x2 = x[..., half_dims:]      # 虚部

        # 6. 进行复数乘法（旋转）
        #    (x1 + i*x2) * (cosθ + i*sinθ) = (x1*cosθ - x2*sinθ) + i*(x2*cosθ + x1*sinθ)
        real = x1 * cos_pos - x2 * sin_pos
        imag = x2 * cos_pos + x1 * sin_pos

        # 7. 拼回原来的形状
        if self.traditional:
            # 传统方式：stack回(N, S, H, half_dims, 2)，再reshape回(N, S, H, D)
            y = mx.stack([real, imag], axis=-1)
            y = y.reshape(N, S, H, D)
        else:
            # 现代方式：直接concat到最后一维
            y = mx.concat([real, imag], axis=-1)
            y = y.reshape(N, S, H, D)

        # 8. 返回与输入相同dtype的结果
        return y.astype(x.dtype)

# 总结：
#   - RoPE的本质是将位置信息编码为复数旋转，嵌入到特征向量中。
#   - 这样做能让自注意力机制天然感知相对位置信息，且实现高效、无须显式加法。
#   - 代码实现严格遵循论文公式，支持多种offset和两种实现风格。
