import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.max_seq_len = max_seq_len
        self.theta = theta
        # head_dim
        self.head_dim = hidden_size // num_heads
        # RoPE编码器
        self.rope = RoPE(self.head_dim, max_seq_len, base=theta)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # 该方法实现了 Qwen2 的多头注意力前向过程，输入 x 形状为 (B, L, E)
        # B: batch size, L: 序列长度, E: embedding 维度

        # 1. 解析各维度
        B, L, E = x.shape
        H_q = self.num_heads         # query 的 head 数
        H = self.num_kv_heads        # key/value 的 head 数（可小于 H_q，支持分组注意力）
        D = self.head_dim            # 每个 head 的维度

        # 2. 线性变换 + bias，得到 q, k, v
        #   - q: (B, L, H_q * D)
        #   - k, v: (B, L, H * D)
        q = linear(x, self.wq, self.bq)
        k = linear(x, self.wk, self.bk)
        v = linear(x, self.wv, self.bv)

        # 3. reshape 拆分 head 维度
        #   - q: (B, L, H_q, D)
        #   - k, v: (B, L, H, D)
        q = q.reshape(B, L, H_q, D)
        k = k.reshape(B, L, H, D)
        v = v.reshape(B, L, H, D)

        # 4. 对 q, k 应用 RoPE 位置编码
        #    这里 offset=slice(offset, offset+L) 表示当前 token 在全局序列中的绝对位置
        #    RoPE 作用后形状不变
        q = self.rope(q, offset=slice(offset, offset + L))  # (B, L, H_q, D)
        k = self.rope(k, offset=slice(offset, offset + L))  # (B, L, H, D)

        # 5. 转置 head 和 seq 维度，方便后续 attention
        #   - q: (B, H_q, L, D)
        #   - k, v: (B, H, L, D)
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))

        # 6. attention 计算（提升到 float32 精度以保证数值稳定性）
        #    - scaled_dot_product_attention_grouped 支持分组 attention
        #    - mask 支持 None/字符串/显式 mask
        q_fp32 = q.astype(mx.float32)
        k_fp32 = k.astype(mx.float32)
        v_fp32 = v.astype(mx.float32)
        attn_out = scaled_dot_product_attention_grouped(
            q_fp32, k_fp32, v_fp32, scale=None, mask=mask
        )  # (B, H_q, L, D)
        attn_out = attn_out.astype(q.dtype)  # 恢复原始精度

        # 7. 恢复输出 shape
        #   - 先转回 (B, L, H_q, D)
        #   - 再合并 head 维度为 (B, L, E)
        attn_out = mx.transpose(attn_out, axes=(0, 2, 1, 3))  # (B, L, H_q, D)
        attn_out = attn_out.reshape(B, L, H_q * D)            # (B, L, E)

        # 8. 输出线性投影
        out = linear(attn_out, self.wo)  # (B, L, E)
        return out


class Qwen2MLP:
    """
    Qwen2 MLP 层实现（无 bias 版本）

    【在 Transformer 里的作用：】
    - MLP（多层感知机）是 Transformer Block 的核心子层之一，通常位于自注意力（Self-Attention）之后。
    - 其主要作用是对每个 token 的表示做逐位置的非线性变换和特征提升，增强模型的表达能力。
    - 具体来说，MLP 通过升维（up projection）、非线性激活、门控机制（Gated Linear Unit, GLU）和降维（down projection）等步骤，帮助模型捕捉更复杂的特征和关系。
    - 在 Qwen2 这类大模型中，MLP 采用了门控结构（GLU），即用 SiLU 激活后的门控向量对 up 投影结果进行逐元素调制，提升了模型的非线性和建模能力。
    - 这种结构能有效提升模型性能，是现代 Transformer 架构的标配。

    结构说明：
    - 输入: x, shape (..., L, E)
    - 1. gate 投影: x @ w_gate.T, shape (..., L, I)
    - 2. up 投影:   x @ w_up.T,   shape (..., L, I)
    - 3. gate 激活: SiLU(gate_out)
    - 4. 门控乘法:  silu(gate_out) * up_out, shape (..., L, I)
    - 5. down 投影: (门控结果) @ w_down.T, shape (..., L, E)
    - 输出: (..., L, E)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        """
        参数:
            dim: 输入/输出维度 E
            hidden_dim: 中间层维度 I
            w_gate: (I, E)
            w_up: (I, E)
            w_down: (E, I)
        """
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate  # (I, E)
        self.w_up = w_up      # (I, E)
        self.w_down = w_down  # (E, I)

    def __call__(self, x: mx.array) -> mx.array:
        """
        前向计算
        输入:
            x: (..., L, E)
        输出:
            (..., L, E)
        """
        # 1. gate 投影: (..., L, I)
        gate_out = mx.matmul(x, self.w_gate.T)
        # 2. up 投影: (..., L, I)
        up_out = mx.matmul(x, self.w_up.T)
        # 3. SiLU 激活
        gate_act = gate_out * mx.sigmoid(gate_out)
        # 4. 门控乘法
        gated = gate_act * up_out
        # 5. down 投影: (..., L, E)
        out = mx.matmul(gated, self.w_down.T)
        return out


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        # 保存参数
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.max_seq_len = max_seq_len
        self.theta = theta

        # LayerNorm 权重
        self.input_layernorm = RMSNorm(
            hidden_size, w_input_layernorm, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps)

        # Attention 权重
        self.attn = Qwen2MultiHeadAttention(
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta,
        )

        # MLP 权重
        self.mlp = Qwen2MLP(
            hidden_size,
            intermediate_size,
            w_gate,
            w_up,
            w_down,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # 1. input_layernorm
        normed_x = self.input_layernorm(x)
        # 2. Attention
        attn_out = self.attn(normed_x, offset, mask)
        # 3. 残差 Add
        x = x + attn_out
        # 4. post_attention_layernorm
        normed_x2 = self.post_attention_layernorm(x)
        # 5. MLP
        mlp_out = self.mlp(normed_x2)
        # 6. 残差 Add
        out = x + mlp_out
        return out


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        pass
