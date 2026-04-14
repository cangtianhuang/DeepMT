"""应用层基准注册表：定义代表性应用场景。

当前覆盖两个代表性方向（Phase J 边界）：
  1. 图像分类（ImageClassification）：测试 CV 模型/应用对图像变换的稳定性
  2. 文本情感分析（TextSentiment）：测试 NLP 模型/应用对文本变换的稳定性

样例输入采用纯 Python 数据结构（list/str），不依赖 numpy/torch：
  - 图像：输入为归一化到 [0, 1] 的浮点列表（扁平化，代表像素值）
  - 文本：输入为 Python 字符串

用法::

    registry = ApplicationBenchmarkRegistry()
    scenarios = registry.list_scenarios()
    sc = registry.get("ImageClassification")
    sc = registry.get("TextSentiment")
"""

from typing import List, Optional

from deepmt.mr_generator.application.scenario import ApplicationScenario
from deepmt.core.logger import logger
from deepmt.ir import ApplicationIR

# ── 场景元数据定义 ────────────────────────────────────────────────────────────

_IMAGE_CLASSIFICATION_FACTS = [
    "图像分类模型将输入图像映射到离散类别标签（如猫/狗/汽车等）。",
    "深度学习图像分类模型对小幅度亮度/对比度变化通常具有较强的鲁棒性。",
    "水平翻转对于不含方向性语义的图像（如动物）通常不改变类别预测。",
    "添加均匀分布的微小噪声（标准差 < 0.05）通常不改变模型的 top-1 预测。",
    "对图像进行归一化偏移（整体平移 ±0.1 范围内）通常不改变类别预测。",
    "图像亮度缩放（乘以 0.8~1.2）对分类结果影响通常较小。",
    "将 RGB 图像的通道顺序改变（如 RGB→BGR）不改变图像内容，但可能影响预期特定通道顺序的模型。",
    "图像分辨率对模型有影响，但在模型内部经过自适应池化后，输入尺寸的小变化影响有限。",
    "模型在训练时使用数据增强，因此对常见的几何/颜色变换具有一定不变性。",
]

_IMAGE_CLASSIFICATION_TRANSFORMS = [
    {
        "name": "brightness_reduce",
        "description": "将图像像素值缩小至 0.9 倍（轻度降低亮度），类别预测应保持不变",
        "transform_code": "lambda s: {**s, 'input': [x * 0.9 for x in s['input']]}",
        "oracle_expr": "label_consistent",
        "category": "noise_robustness",
        "risk_level": "low",
        "rationale": "轻度降低亮度不改变图像的语义内容，分类模型在训练中通常见过亮度变化",
    },
    {
        "name": "brightness_increase",
        "description": "将图像像素值放大至 1.1 倍并裁剪到 [0,1]，类别预测应保持不变",
        "transform_code": "lambda s: {**s, 'input': [min(1.0, x * 1.1) for x in s['input']]}",
        "oracle_expr": "label_consistent",
        "category": "noise_robustness",
        "risk_level": "low",
        "rationale": "轻度增加亮度不改变图像语义，裁剪到合法范围后仍为有效图像",
    },
    {
        "name": "contrast_reduce",
        "description": "降低图像对比度（像素值向均值靠拢），类别预测应保持不变",
        "transform_code": "lambda s: {**s, 'input': [(x - 0.5) * 0.8 + 0.5 for x in s['input']]}",
        "oracle_expr": "label_consistent",
        "category": "noise_robustness",
        "risk_level": "low",
        "rationale": "轻度降低对比度保留了图像的主要结构特征，语义内容不变",
    },
    {
        "name": "additive_noise_tiny",
        "description": "向像素值添加微小常数偏移（+0.01），类别预测应保持不变",
        "transform_code": "lambda s: {**s, 'input': [min(1.0, max(0.0, x + 0.01)) for x in s['input']]}",
        "oracle_expr": "label_consistent",
        "category": "noise_robustness",
        "risk_level": "low",
        "rationale": "极小的整体像素偏移对人类和模型均不改变感知类别",
    },
    {
        "name": "normalize_shift",
        "description": "对图像像素值进行整体平移使均值接近 0（中心化），类别预测应保持不变",
        "transform_code": "lambda s: {**s, 'input': [x - (sum(s['input']) / len(s['input'])) for x in s['input']]}",
        "oracle_expr": "label_consistent",
        "category": "invariance",
        "risk_level": "medium",
        "rationale": "许多图像分类模型在归一化前后的输入上均可正确预测，但依赖绝对亮度值的模型可能受影响",
    },
]

_TEXT_SENTIMENT_FACTS = [
    "情感分析任务将文本映射到情感极性（正面/负面/中性），或 1-5 星评分。",
    "文本的大小写通常不影响情感极性，'Great product' 和 'great product' 应有相同情感。",
    "去除标点符号通常不改变文本的整体情感倾向（句子结构保持不变时）。",
    "去除多余空白字符（两端空格、连续空格）不改变文本语义，情感预测应相同。",
    "在句子开头添加情感中性的描述词（如 'I think that '）通常不改变原文的情感极性。",
    "复述同一内容（如将文本重复两遍并用句号分隔）通常不改变情感极性，但置信度可能更高。",
    "将文本中的数字用文字替换（如 '5 stars' → 'five stars'）通常不改变情感极性。",
    "情感分析对词序敏感，打乱词序会显著改变语义，应避免词序打乱变换。",
    "替换关键情感词（如将 'good' 换成 'excellent'）可能加强情感强度，但不改变极性方向。",
    "对评论文本添加无情感倾向的附属信息（如 'Review: ...'）通常不改变核心情感极性。",
]

_TEXT_SENTIMENT_TRANSFORMS = [
    {
        "name": "lowercase",
        "description": "将文本转为全小写，情感极性预测应保持不变",
        "transform_code": "lambda s: {**s, 'text': s['text'].lower()}",
        "oracle_expr": "label_consistent",
        "category": "invariance",
        "risk_level": "low",
        "rationale": "情感分析通常对大小写不敏感，语义内容不变",
    },
    {
        "name": "strip_whitespace",
        "description": "去除文本两端空白字符，情感极性预测应保持不变",
        "transform_code": "lambda s: {**s, 'text': s['text'].strip()}",
        "oracle_expr": "label_consistent",
        "category": "invariance",
        "risk_level": "low",
        "rationale": "两端空格是格式噪声，不影响文本语义内容",
    },
    {
        "name": "normalize_spaces",
        "description": "将连续空格规范化为单个空格，情感极性预测应保持不变",
        "transform_code": "lambda s: {**s, 'text': ' '.join(s['text'].split())}",
        "oracle_expr": "label_consistent",
        "category": "invariance",
        "risk_level": "low",
        "rationale": "空格规范化仅影响格式，不改变词序或情感内容",
    },
    {
        "name": "add_neutral_prefix",
        "description": "在文本前添加中性前缀 'Review: '，情感极性预测应保持不变",
        "transform_code": "lambda s: {**s, 'text': 'Review: ' + s['text']}",
        "oracle_expr": "label_consistent",
        "category": "semantic_preservation",
        "risk_level": "low",
        "rationale": "添加中性标签前缀不引入情感信息，不改变原文情感极性",
    },
    {
        "name": "uppercase",
        "description": "将文本转为全大写，情感极性预测应保持不变",
        "transform_code": "lambda s: {**s, 'text': s['text'].upper()}",
        "oracle_expr": "label_consistent",
        "category": "invariance",
        "risk_level": "low",
        "rationale": "大写通常用于强调，不改变情感极性（部分模型可能视全大写为更强烈语气）",
    },
]

# 样例输入（纯 Python，不依赖 numpy/torch）
_IMAGE_SAMPLE_INPUTS = [
    # 每个样例是一个 dict，'input' 为归一化到 [0,1] 的 16 个像素值（4x4 灰度图的扁平化）
    {"input": [0.9, 0.8, 0.7, 0.6, 0.8, 0.9, 0.8, 0.7, 0.7, 0.8, 0.9, 0.8, 0.6, 0.7, 0.8, 0.9]},  # 亮色
    {"input": [0.1, 0.2, 0.3, 0.4, 0.2, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.2, 0.4, 0.3, 0.2, 0.1]},  # 暗色
    {"input": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]},  # 均匀
    {"input": [0.9, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.9]},  # 棋盘
]
_IMAGE_SAMPLE_LABELS = [1, 0, 1, 0]  # 亮色=类别1, 暗色=类别0（mock 规则）

_TEXT_SAMPLE_INPUTS = [
    {"text": "This product is absolutely amazing! I love it."},
    {"text": "Terrible experience. Would not recommend to anyone."},
    {"text": "It was okay, nothing special but not bad either."},
    {"text": "Great quality and fast shipping! Very satisfied."},
    {"text": "Completely disappointed. The item broke after one day."},
]
_TEXT_SAMPLE_LABELS = [1, 0, 1, 1, 0]  # 1=positive, 0=negative/neutral

# ── 场景注册表 ─────────────────────────────────────────────────────────────────

_SCENARIO_SPECS = [
    ApplicationScenario(
        name="ImageClassification",
        task_type="image_classification",
        domain="computer_vision",
        input_type="image_array",
        output_type="class_label",
        input_schema="归一化到 [0,1] 的像素浮点列表（dict 含 'input' 键）",
        output_schema="整数类别标签（如 0/1 表示暗/亮）",
        description=(
            "图像分类应用：输入图像数组，输出整数类别标签。"
            "测试模型对常见图像变换（亮度、噪声等）的稳定性。"
        ),
        domain_facts=_IMAGE_CLASSIFICATION_FACTS,
        sample_inputs=_IMAGE_SAMPLE_INPUTS,
        sample_labels=_IMAGE_SAMPLE_LABELS,
        known_transforms=_IMAGE_CLASSIFICATION_TRANSFORMS,
    ),
    ApplicationScenario(
        name="TextSentiment",
        task_type="text_sentiment",
        domain="nlp",
        input_type="text_string",
        output_type="sentiment_label",
        input_schema="Python 字符串（dict 含 'text' 键），自然语言文本",
        output_schema="整数情感标签（0=负面/中性, 1=正面）",
        description=(
            "文本情感分析应用：输入自然语言文本，输出情感极性标签（正面/负面）。"
            "测试模型对文本格式变换（大小写、空格等）的稳定性。"
        ),
        domain_facts=_TEXT_SENTIMENT_FACTS,
        sample_inputs=_TEXT_SAMPLE_INPUTS,
        sample_labels=_TEXT_SAMPLE_LABELS,
        known_transforms=_TEXT_SENTIMENT_TRANSFORMS,
    ),
]


class ApplicationBenchmarkRegistry:
    """应用层基准场景注册表。

    提供所有预定义应用场景的 ApplicationScenario 及对应 ApplicationIR。

    用法::

        registry = ApplicationBenchmarkRegistry()
        names = registry.names()
        sc = registry.get("ImageClassification")
        ir = registry.get_ir("TextSentiment")
    """

    def list_scenarios(self) -> List[ApplicationScenario]:
        """返回所有基准场景列表。"""
        return list(_SCENARIO_SPECS)

    def get(self, name: str) -> Optional[ApplicationScenario]:
        """按名称获取场景。"""
        for sc in _SCENARIO_SPECS:
            if sc.name == name:
                return sc
        logger.warning(f"[AppRegistry] 未找到场景: {name!r}")
        return None

    def names(self) -> List[str]:
        """返回所有场景名称列表。"""
        return [sc.name for sc in _SCENARIO_SPECS]

    def get_ir(self, name: str) -> Optional["ApplicationIR"]:
        """将场景转换为 ApplicationIR 对象并返回。"""
        from deepmt.ir import ApplicationIR

        sc = self.get(name)
        if sc is None:
            return None
        return ApplicationIR(
            name=sc.name,
            framework=None,
            task_type=sc.task_type,
            domain=sc.domain,
            input_description=sc.input_schema,
            output_description=sc.output_schema,
            sample_inputs=sc.sample_inputs,
            sample_labels=sc.sample_labels,
        )
