"""模型层变换策略库：为模型层 MR 生成提供可复用的输入变换策略。

每个策略描述了：
  - 一种对模型输入的变换方式（transform_code）
  - 对应期望的 oracle 类型（oracle_type）
  - 策略的适用条件（applicable_to）

策略设计原则：
  - 变换只作用于输入张量，不修改模型权重
  - transform_code 是一个 lambda 字符串：lambda x: ...
    其中 x 是输入张量（float 或 int，视模型类型而定）
  - oracle_type 是 ModelVerifier 可识别的字符串常量

支持的 oracle 类型（由 ModelVerifier 实现）：
  - "prediction_consistent"  argmax(orig) == argmax(trans)
  - "topk_consistent"        top-k 预测集合相同
  - "output_close"           outputs allclose within tolerance
  - "output_order"           相对概率排序不变

注意：变换对整数输入类型（embedding 输入）仅支持部分策略。
"""

from dataclasses import dataclass, field
from typing import List, Optional

from deepmt.mr_generator.model.graph_analyzer import ModelAnalysisResult


# ── 策略数据结构 ──────────────────────────────────────────────────────────────


@dataclass
class TransformStrategy:
    """单条变换策略描述。

    Attributes:
        name:           策略唯一名称（如 "input_scale_2x"）
        category:       变换类别（"input_transform"）
        transform_code: lambda 字符串，形如 "lambda x: ..."
        oracle_type:    期望的 oracle 类型（ModelVerifier 可识别）
        oracle_params:  oracle 额外参数（如 tolerance、k 值）
        description:    人类可读描述
        applicable_to:  适用的模型类型列表；None 表示通用
        input_dtype:    仅对此 dtype 输入适用（"float" / "int" / None=两者均可）
    """

    name: str
    category: str
    transform_code: str
    oracle_type: str
    oracle_params: dict = field(default_factory=dict)
    description: str = ""
    applicable_to: Optional[List[str]] = None   # ["mlp", "cnn", ...] or None
    input_dtype: Optional[str] = None           # "float" | "int" | None


# ── 策略库定义 ────────────────────────────────────────────────────────────────

# 注：transform_code 中的 x 是输入张量（batch 维度在外）
# 对于 float 输入（MLP/CNN）：x.shape = (batch, *input_shape)
# 对于 int 输入（RNN/Transformer）：x.shape = (batch, seq_len)，dtype=int64

_ALL_STRATEGIES: List[TransformStrategy] = [
    # ── 通用 float 变换 ──────────────────────────────────────────────────────
    TransformStrategy(
        name="input_scale_small",
        category="input_transform",
        transform_code="lambda x: x * 0.5",
        oracle_type="prediction_consistent",
        description="将输入缩小为原来的 0.5 倍，预测类别应保持不变（分类任务的尺度不变性）",
        applicable_to=["mlp", "cnn"],
        input_dtype="float",
    ),
    TransformStrategy(
        name="input_scale_large",
        category="input_transform",
        transform_code="lambda x: x * 2.0",
        oracle_type="prediction_consistent",
        description="将输入放大 2 倍，预测类别应保持不变（分类尺度不变性，较强假设）",
        applicable_to=["mlp", "cnn"],
        input_dtype="float",
    ),
    TransformStrategy(
        name="input_additive_noise_tiny",
        category="input_transform",
        transform_code=(
            "lambda x: x + __import__('torch').randn_like(x) * 1e-4"
        ),
        oracle_type="prediction_consistent",
        oracle_params={"tolerance": 1e-4},
        description="添加极小高斯噪声（σ=1e-4），预测应保持一致（鲁棒性最小验证）",
        applicable_to=["mlp", "cnn"],
        input_dtype="float",
    ),
    TransformStrategy(
        name="input_additive_noise_small",
        category="input_transform",
        transform_code=(
            "lambda x: x + __import__('torch').randn_like(x) * 1e-2"
        ),
        oracle_type="prediction_consistent",
        oracle_params={"tolerance": 1e-2},
        description="添加小高斯噪声（σ=0.01），预测应保持一致（一般鲁棒性验证）",
        applicable_to=["mlp", "cnn"],
        input_dtype="float",
    ),
    TransformStrategy(
        name="batch_order_invariance",
        category="input_transform",
        transform_code="lambda x: x.flip(0)",
        oracle_type="output_order_invariant",
        description="翻转 batch 顺序，每个样本的输出应独立且顺序颠倒（batch 无关性验证）",
        applicable_to=["mlp", "cnn"],
        input_dtype="float",
    ),
    # ── CNN 专用 ──────────────────────────────────────────────────────────────
    TransformStrategy(
        name="spatial_flip_horizontal",
        category="input_transform",
        transform_code="lambda x: x.flip(-1)",
        oracle_type="prediction_consistent",
        description="水平翻转图像（最后维度），分类预测应保持（水平对称性验证）",
        applicable_to=["cnn"],
        input_dtype="float",
    ),
    TransformStrategy(
        name="spatial_flip_vertical",
        category="input_transform",
        transform_code="lambda x: x.flip(-2)",
        oracle_type="prediction_consistent",
        description="垂直翻转图像（倒数第二维度），分类预测应保持（垂直对称性验证）",
        applicable_to=["cnn"],
        input_dtype="float",
    ),
    TransformStrategy(
        name="channel_scale",
        category="input_transform",
        transform_code="lambda x: x * 1.0",
        oracle_type="output_close",
        oracle_params={"atol": 1e-5},
        description="乘以 1.0（恒等变换），输出应精确不变（基准一致性验证）",
        applicable_to=["cnn"],
        input_dtype="float",
    ),
    # ── 整数输入（RNN/Transformer）─────────────────────────────────────────────
    TransformStrategy(
        name="sequence_identity",
        category="input_transform",
        transform_code="lambda x: x.clone()",
        oracle_type="output_close",
        oracle_params={"atol": 1e-6},
        description="复制序列（恒等变换），输出应完全一致（基准一致性）",
        applicable_to=["rnn", "transformer"],
        input_dtype="int",
    ),
    TransformStrategy(
        name="sequence_batch_flip",
        category="input_transform",
        transform_code="lambda x: x.flip(0)",
        oracle_type="output_order_invariant",
        description="翻转 batch 内样本顺序，各样本输出应独立（batch 无关性验证）",
        applicable_to=["rnn", "transformer"],
        input_dtype="int",
    ),
    TransformStrategy(
        name="sequence_prediction_consistent",
        category="input_transform",
        transform_code="lambda x: x.clone()",
        oracle_type="prediction_consistent",
        description="恒等序列变换，预测类别应保持一致（Transformer 确定性验证）",
        applicable_to=["transformer"],
        input_dtype="int",
    ),
    # ── 通用恒等变换（所有架构）────────────────────────────────────────────────
    TransformStrategy(
        name="identity_float",
        category="input_transform",
        transform_code="lambda x: x + 0.0",
        oracle_type="output_close",
        oracle_params={"atol": 1e-6},
        description="恒等变换（+0），输出应数值完全一致（数值稳定性基线）",
        applicable_to=["mlp", "cnn"],
        input_dtype="float",
    ),
]

# 按名称建立快速查找索引
_STRATEGY_INDEX = {s.name: s for s in _ALL_STRATEGIES}


# ── 策略选择器 ────────────────────────────────────────────────────────────────


class TransformStrategyLibrary:
    """变换策略库：根据模型分析结果选择适用策略。

    用法::

        library = TransformStrategyLibrary()
        strategies = library.select(analysis_result)
        for s in strategies:
            print(s.name, s.transform_code, s.oracle_type)
    """

    def select(
        self,
        analysis: ModelAnalysisResult,
        max_strategies: Optional[int] = None,
    ) -> List[TransformStrategy]:
        """根据模型结构分析结果返回适用的变换策略列表。

        Args:
            analysis:       ModelAnalysisResult 实例
            max_strategies: 最多返回的策略数；None 表示返回全部适用策略

        Returns:
            适用策略列表，按优先级排序
        """
        model_type = analysis.model_type
        input_dtype = "int" if analysis.input_is_integer else "float"

        selected = []
        for strategy in _ALL_STRATEGIES:
            # 检查模型类型适用性
            if strategy.applicable_to is not None:
                if model_type not in strategy.applicable_to:
                    continue
            # 检查输入 dtype 适用性
            if strategy.input_dtype is not None:
                if strategy.input_dtype != input_dtype:
                    continue
            selected.append(strategy)

        if max_strategies is not None:
            selected = selected[:max_strategies]

        return selected

    def get(self, name: str) -> Optional[TransformStrategy]:
        """按名称获取策略；未找到返回 None。"""
        return _STRATEGY_INDEX.get(name)

    def all_strategies(self) -> List[TransformStrategy]:
        """返回全部策略列表。"""
        return list(_ALL_STRATEGIES)
