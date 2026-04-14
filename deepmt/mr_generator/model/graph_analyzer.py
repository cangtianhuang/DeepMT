"""模型结构分析器：从 PyTorch 模型提取结构信息，为 MR 生成提供结构输入。

分析流程：
  1. 遍历模型的 named_modules，提取所有叶子层的类型序列
  2. 识别关键组件标志（conv、rnn、attention、norm、residual）
  3. 推断模型类型（mlp / cnn / rnn / transformer）
  4. 计算参数量与深度
  5. 输出 ModelAnalysisResult 供 MR 生成器消费

接口设计原则（插件职责边界）：
  - 本模块仅负责结构静态分析，不执行任何前向推理
  - 不持有框架插件引用，通过 `nn.Module` 原生接口操作
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from deepmt.core.logger import logger
from deepmt.ir import ModelIR

# ── 已知层类型到语义标签的映射 ──────────────────────────────────────────────────

# 每个 key 是 nn.Module 子类名前缀或精确名；value 是语义标签
_LAYER_TAGS: Dict[str, str] = {
    # 线性层
    "Linear": "linear",
    "Bilinear": "linear",
    # 卷积
    "Conv1d": "conv",
    "Conv2d": "conv",
    "Conv3d": "conv",
    "ConvTranspose1d": "conv",
    "ConvTranspose2d": "conv",
    "ConvTranspose3d": "conv",
    # 池化
    "MaxPool1d": "pool",
    "MaxPool2d": "pool",
    "MaxPool3d": "pool",
    "AvgPool1d": "pool",
    "AvgPool2d": "pool",
    "AvgPool3d": "pool",
    "AdaptiveAvgPool1d": "pool",
    "AdaptiveAvgPool2d": "pool",
    "AdaptiveAvgPool3d": "pool",
    "AdaptiveMaxPool2d": "pool",
    # 循环层
    "RNN": "rnn",
    "LSTM": "rnn",
    "GRU": "rnn",
    # 注意力
    "MultiheadAttention": "attention",
    "TransformerEncoderLayer": "attention",
    "TransformerDecoderLayer": "attention",
    "TransformerEncoder": "attention",
    "TransformerDecoder": "attention",
    # 归一化
    "BatchNorm1d": "norm",
    "BatchNorm2d": "norm",
    "BatchNorm3d": "norm",
    "LayerNorm": "norm",
    "GroupNorm": "norm",
    "InstanceNorm1d": "norm",
    "InstanceNorm2d": "norm",
    # 激活
    "ReLU": "activation",
    "LeakyReLU": "activation",
    "ELU": "activation",
    "GELU": "activation",
    "Sigmoid": "activation",
    "Tanh": "activation",
    "Softmax": "activation",
    "LogSoftmax": "activation",
    "SiLU": "activation",
    # Dropout
    "Dropout": "dropout",
    "Dropout2d": "dropout",
    "AlphaDropout": "dropout",
    # 嵌入
    "Embedding": "embedding",
    "EmbeddingBag": "embedding",
    # 展平
    "Flatten": "flatten",
}


def _get_tag(module_class_name: str) -> str:
    """将模块类名映射到语义标签，未知类型返回 'other'。"""
    return _LAYER_TAGS.get(module_class_name, "other")


# ── 分析结果数据结构 ───────────────────────────────────────────────────────────


@dataclass
class LayerInfo:
    """单个叶子层的描述。"""

    path: str           # 层在模型中的路径，如 "conv1" / "encoder.layers.0.self_attn"
    class_name: str     # nn.Module 子类名，如 "Conv2d"
    tag: str            # 语义标签，如 "conv"
    params: int         # 该层可训练参数量
    extra: Dict[str, Any] = field(default_factory=dict)  # 层特有属性（kernel_size 等）


@dataclass
class ModelAnalysisResult:
    """模型结构分析结果，供 MR 生成器消费。

    Attributes:
        model_name:       模型名称
        model_type:       推断的模型类型（"mlp"/"cnn"/"rnn"/"transformer"/"unknown"）
        task_type:        任务类型（由外部传入 or 推断）
        layer_infos:      叶子层列表（按前序遍历顺序）
        layer_tags:       layer_infos 中每层 tag 的序列（便于快速模式匹配）
        has_conv:         是否包含卷积层
        has_rnn:          是否包含循环层
        has_attention:    是否包含注意力层
        has_normalization: 是否包含归一化层
        has_embedding:    是否包含嵌入层
        input_is_integer: 输入是否为整数（embedding 场景）
        num_parameters:   全模型可训练参数量
        depth:            叶子层数（不含容器）
        key_patterns:     识别到的关键子结构模式列表，如 ["conv_pool", "attention_block"]
        input_shape:      单样本输入形状（由外部提供 or None）
        output_shape:     单样本输出形状（由外部提供 or None）
        num_classes:      分类类别数（由外部提供 or None）
    """

    model_name: str
    model_type: str
    task_type: str
    layer_infos: List[LayerInfo]
    layer_tags: List[str]
    has_conv: bool
    has_rnn: bool
    has_attention: bool
    has_normalization: bool
    has_embedding: bool
    input_is_integer: bool
    num_parameters: int
    depth: int
    key_patterns: List[str]
    input_shape: Optional[Tuple] = None
    output_shape: Optional[Tuple] = None
    num_classes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典，供 ModelIR.analysis_summary 存储。"""
        return {
            "model_type": self.model_type,
            "task_type": self.task_type,
            "has_conv": self.has_conv,
            "has_rnn": self.has_rnn,
            "has_attention": self.has_attention,
            "has_normalization": self.has_normalization,
            "has_embedding": self.has_embedding,
            "input_is_integer": self.input_is_integer,
            "num_parameters": self.num_parameters,
            "depth": self.depth,
            "key_patterns": self.key_patterns,
            "layer_tags": self.layer_tags,
        }


# ── 分析器主类 ────────────────────────────────────────────────────────────────


class ModelGraphAnalyzer:
    """模型结构分析器。

    使用 PyTorch nn.Module 的 named_modules 接口静态提取结构信息，
    不执行前向推理，不依赖输入数据。

    用法::

        analyzer = ModelGraphAnalyzer()
        result = analyzer.analyze(model_ir)
        print(result.model_type, result.key_patterns)
    """

    def analyze(self, model_ir: ModelIR) -> ModelAnalysisResult:
        """分析 ModelIR 对象，返回结构分析结果。

        Args:
            model_ir: 已填充 model_instance 的 ModelIR

        Returns:
            ModelAnalysisResult
        """
        model = model_ir.model_instance
        if model is None:
            raise ValueError(
                f"[GraphAnalyzer] ModelIR '{model_ir.name}' 的 model_instance 为 None，"
                "请先通过 ModelBenchmarkRegistry.get(with_instance=True) 获取"
            )

        layer_infos = self._extract_leaves(model)
        layer_tags = [li.tag for li in layer_infos]

        # 结构特征检测：扫描所有模块（含非叶子），以捕获 LSTM/MultiheadAttention 等
        # 在新版 PyTorch 中本身有子模块的结构层（如 MultiheadAttention.out_proj）
        all_class_names = {type(m).__name__ for _, m in model.named_modules() if _}
        all_tags = {_get_tag(cn) for cn in all_class_names}

        has_conv = "conv" in all_tags
        has_rnn = "rnn" in all_tags
        has_attention = "attention" in all_tags
        has_norm = "norm" in all_tags or "norm" in layer_tags
        has_embedding = "embedding" in all_tags or "embedding" in layer_tags
        input_is_integer = has_embedding

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        depth = len(layer_infos)

        model_type = self._infer_model_type(
            has_conv, has_rnn, has_attention, has_embedding, layer_tags
        )
        task_type = model_ir.task_type or "classification"
        key_patterns = self._extract_patterns(layer_tags, has_conv, has_rnn, has_attention)

        result = ModelAnalysisResult(
            model_name=model_ir.name,
            model_type=model_type,
            task_type=task_type,
            layer_infos=layer_infos,
            layer_tags=layer_tags,
            has_conv=has_conv,
            has_rnn=has_rnn,
            has_attention=has_attention,
            has_normalization=has_norm,
            has_embedding=has_embedding,
            input_is_integer=input_is_integer,
            num_parameters=num_params,
            depth=depth,
            key_patterns=key_patterns,
            input_shape=model_ir.input_shape,
            output_shape=model_ir.output_shape,
            num_classes=model_ir.num_classes,
        )

        logger.debug(
            f"[GraphAnalyzer] {model_ir.name}: type={model_type}, "
            f"depth={depth}, params={num_params}, patterns={key_patterns}"
        )
        return result

    def analyze_and_fill(self, model_ir: ModelIR) -> ModelAnalysisResult:
        """分析并将结果回填到 model_ir.analysis_summary。"""
        result = self.analyze(model_ir)
        model_ir.analysis_summary = result.to_dict()
        model_ir.model_type = result.model_type
        if not model_ir.task_type:
            model_ir.task_type = result.task_type
        return result

    # ── 私有方法 ──────────────────────────────────────────────────────────────

    def _extract_leaves(self, model: Any) -> List[LayerInfo]:
        """提取模型所有叶子层（无子模块的 nn.Module）。"""
        leaves = []
        for path, module in model.named_modules():
            if not path:  # 跳过根模块自身
                continue
            # 判断是否为叶子层（children 为空）
            if sum(1 for _ in module.children()) > 0:
                continue

            class_name = type(module).__name__
            tag = _get_tag(class_name)
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            extra = self._extract_extra(module, class_name)

            leaves.append(LayerInfo(
                path=path,
                class_name=class_name,
                tag=tag,
                params=params,
                extra=extra,
            ))
        return leaves

    def _extract_extra(self, module: Any, class_name: str) -> Dict[str, Any]:
        """提取层的关键超参数（用于 MR 生成参考）。"""
        extra: Dict[str, Any] = {}
        try:
            if class_name in ("Conv1d", "Conv2d", "Conv3d"):
                extra["in_channels"] = module.in_channels
                extra["out_channels"] = module.out_channels
                extra["kernel_size"] = module.kernel_size
            elif class_name == "Linear":
                extra["in_features"] = module.in_features
                extra["out_features"] = module.out_features
            elif class_name in ("RNN", "LSTM", "GRU"):
                extra["input_size"] = module.input_size
                extra["hidden_size"] = module.hidden_size
                extra["num_layers"] = module.num_layers
            elif class_name == "MultiheadAttention":
                extra["embed_dim"] = module.embed_dim
                extra["num_heads"] = module.num_heads
            elif class_name == "Embedding":
                extra["num_embeddings"] = module.num_embeddings
                extra["embedding_dim"] = module.embedding_dim
        except AttributeError:
            pass
        return extra

    def _infer_model_type(
        self,
        has_conv: bool,
        has_rnn: bool,
        has_attention: bool,
        has_embedding: bool,
        tags: List[str],
    ) -> str:
        """基于组件标志推断模型架构类型。"""
        if has_attention:
            return "transformer"
        if has_rnn:
            return "rnn"
        if has_conv:
            return "cnn"
        linear_count = tags.count("linear")
        if linear_count >= 2:
            return "mlp"
        return "unknown"

    def _extract_patterns(
        self,
        tags: List[str],
        has_conv: bool,
        has_rnn: bool,
        has_attention: bool,
    ) -> List[str]:
        """识别有价值的子结构模式，供 MR 生成器选择策略。"""
        patterns: List[str] = []

        # conv + pool 组合（CNN 典型块）
        if has_conv and "pool" in tags:
            patterns.append("conv_pool")

        # conv + norm 组合
        if has_conv and "norm" in tags:
            patterns.append("conv_norm")

        # fc_block：两个以上 linear 层
        if tags.count("linear") >= 2:
            patterns.append("fc_block")

        # attention_block
        if has_attention:
            patterns.append("attention_block")

        # rnn_block
        if has_rnn:
            patterns.append("rnn_block")

        # embedding_input：输入为整数索引
        if "embedding" in tags:
            patterns.append("embedding_input")

        # dropout_regularized
        if "dropout" in tags:
            patterns.append("dropout_regularized")

        return patterns
