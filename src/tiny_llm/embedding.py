import mlx.core as mx


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        '''
        Embedding 层的构造函数
        参数:
            vocab_size: 词表大小
            embedding_dim: 词嵌入维度
            weight: 词嵌入矩阵，形状为 (vocab_size, embedding_dim)
                   每一行代表一个词的嵌入向量
        '''
        self.vocab_size = vocab_size  # 词表大小
        self.embedding_dim = embedding_dim  # 词嵌入维度
        self.weight = weight  # 词嵌入矩阵

    def __call__(self, x: mx.array) -> mx.array:
        '''
        前向传播，将输入的词 id 转换为词嵌入向量
        参数:
            x: 形状为 (...,) 的整数数组，表示词 id
               支持任意前缀维度，但元素必须是 [0, vocab_size) 范围内的整数
        返回:
            形状为 (..., embedding_dim) 的词嵌入向量
            与输入形状相同，但最后增加一维 embedding_dim
        实现:
            使用 mx.take 从词嵌入矩阵中查表得到词向量
            axis=0 表示在第 0 维(词表维度)上索引
        '''
        print('x shape: ', x.shape)
        print('self.weight shape: ', self.weight.shape)
        return mx.take(self.weight, x, axis=0)

    def as_linear(self, x: mx.array) -> mx.array:
        '''
        将 Embedding 层当作线性层使用
        参数:
            x: 形状为 (..., embedding_dim) 的输入向量
        返回:
            形状为 (..., vocab_size) 的输出向量
        实现:
            计算 x @ W^T，其中 W 是词嵌入矩阵
            这在语言模型的输出层中很常用，用于预测下一个词的概率分布
            等价于计算输入向量与每个词嵌入的内积，得到相似度分数
        '''
        return mx.matmul(x, self.weight.T)
