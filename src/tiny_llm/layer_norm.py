import mlx.core as mx


# RMSNorm（Root Mean Square Layer Normalization）是一种替代 LayerNorm 的归一化方法，最早由 Zhang & Sennrich 在论文
# "Root Mean Square Layer Normalization"（https://arxiv.org/abs/1910.07467）中提出。
#
# 论文核心思想：
# - LayerNorm 会对输入 x 的最后一维做均值-方差归一化（即减去均值再除以标准差），而 RMSNorm 只做方差归一化（不减均值，只除以均方根）。
# - 具体公式如下：
#     y = x / RMS(x) * weight
#   其中 RMS(x) = sqrt(mean(x ** 2) + eps)，weight 是可学习参数，eps 是防止除零的小常数。
# - 这样做的好处是：RMSNorm 计算更简单，去掉了均值项，推理和训练速度更快，且在大模型上表现与 LayerNorm 相当甚至更好。
# - RMSNorm 适用于 Transformer 等结构，尤其在大模型（如 GPT-3、Qwen2 等）中被广泛采用。
#
# 作用总结：
# - 提升数值稳定性（归一化防止激活爆炸/消失）
# - 加速训练和推理
# - 简化归一化操作，节省计算资源
# - 在大模型中能取得与 LayerNorm 相当甚至更优的效果

class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        """
        RMSNorm 构造函数
        参数:
            dim: 归一化的最后一维大小
            weight: 可学习的缩放参数，shape=(dim,)
            eps: 防止除零的小常数
        """
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """
        前向计算
        输入:
            x: shape (..., dim)
        输出:
            归一化并缩放后的结果，shape同输入
        """
        # 1. 为了数值稳定，先将输入转为 float32
        x_fp32 = x.astype(mx.float32)
        # 2. 计算最后一维的均方根 RMS(x) = sqrt(mean(x^2) + eps)
        rms = mx.sqrt(mx.mean(x_fp32 * x_fp32, axis=-
                      1, keepdims=True) + self.eps)
        # 3. 归一化（不减均值，只除以 RMS）
        x_norm = x_fp32 / rms
        # 4. 乘以可学习的 weight（先转 float32 以便广播），最后转回原始精度
        out = x_norm * self.weight.astype(mx.float32)
        return out.astype(x.dtype)
