"""Phase I 基准模型定义：SimpleMLP、SimpleCNN、SimpleRNN、TinyTransformer。

这些模型用于模型层蜕变测试的标准基准对象，设计原则：
  - 结构简洁，具有代表性的层组合
  - 参数量小（推理速度快，便于批量测试）
  - 每个模型覆盖一种典型的神经网络架构模式
"""

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError("基准模型需要 PyTorch，请先安装：pip install torch")


# ── SimpleMLP ─────────────────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
    """两层 MLP 分类器。

    结构：Linear → ReLU → Linear → Softmax（推理时）
    输入：(batch, input_dim)
    输出：(batch, num_classes)
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        _require_torch()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ── SimpleCNN ──────────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """两卷积层 + 两全连接层分类 CNN。

    结构：Conv2d → ReLU → MaxPool → Conv2d → ReLU → MaxPool → Flatten → Linear → ReLU → Linear
    输入：(batch, 1, 28, 28)  — 仿 MNIST 尺寸
    输出：(batch, num_classes)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        _require_torch()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.fc1(self.flatten(x)))
        return self.fc2(x)


# ── SimpleRNN ──────────────────────────────────────────────────────────────────

class SimpleRNN(nn.Module):
    """单层 LSTM + 线性头分类器。

    结构：Embedding → LSTM → 取最后时刻隐状态 → Linear
    输入：(batch, seq_len) — 整数索引序列
    输出：(batch, num_classes)
    """

    def __init__(
        self,
        vocab_size: int = 100,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        num_classes: int = 10,
        seq_len: int = 16,
    ):
        super().__init__()
        _require_torch()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.seq_len = seq_len

    def forward(self, x):
        # x: (batch, seq_len) int indices
        emb = self.embedding(x)               # (batch, seq, embed_dim)
        out, _ = self.lstm(emb)               # (batch, seq, hidden_dim)
        last = out[:, -1, :]                  # (batch, hidden_dim)
        return self.fc(last)


# ── TinyTransformer ────────────────────────────────────────────────────────────

class TinyTransformer(nn.Module):
    """单层 Transformer Encoder + 均值池化 + 线性头分类器。

    结构：Embedding → TransformerEncoderLayer → mean-pool → Linear
    输入：(batch, seq_len) — 整数索引序列
    输出：(batch, num_classes)
    """

    def __init__(
        self,
        vocab_size: int = 100,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_classes: int = 10,
        seq_len: int = 16,
    ):
        super().__init__()
        _require_torch()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=64,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.seq_len = seq_len

    def forward(self, x):
        emb = self.embedding(x)             # (batch, seq, embed_dim)
        enc = self.encoder(emb)             # (batch, seq, embed_dim)
        pooled = enc.mean(dim=1)            # (batch, embed_dim)
        return self.fc(pooled)
