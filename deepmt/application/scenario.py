"""应用层场景描述：定义代表性测试场景的结构与领域知识。

一个 ApplicationScenario 包含：
  - 场景元数据（名称、任务类型、领域）
  - 输入/输出格式描述（供 LLM 上下文使用）
  - 领域知识片段（domain_facts）：用于 LLM 推理应用层 MR 的知识依据
  - 样例输入与预期标签（用于验证）

设计原则：
  - 场景知识是静态的领域事实，不依赖网络或 LLM
  - 样例输入足够简单，可在无框架环境下运行（纯 Python）
  - 场景驱动 AppContextBuilder 和 ApplicationLLMMRGenerator
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ApplicationScenario:
    """应用层测试场景描述。

    Attributes:
        name:               场景唯一名称，如 "ImageClassification"
        task_type:          任务类型，如 "image_classification"、"text_sentiment"
        domain:             领域，如 "computer_vision"、"nlp"
        input_type:         输入数据类型描述，如 "image_array"、"text_string"
        output_type:        输出数据类型描述，如 "class_label"、"sentiment_label"
        input_schema:       输入格式描述（自然语言 + 类型）
        output_schema:      输出格式描述（自然语言 + 类型）
        description:        场景整体描述
        domain_facts:       领域知识片段列表（用于 LLM 推理的知识依据）
        sample_inputs:      样例输入列表（dict 格式，含 'input'/'text' 等键）
        sample_labels:      样例预期标签列表
        known_transforms:   已知有效变换描述（可选，用于模板回退）
    """

    name: str
    task_type: str
    domain: str
    input_type: str
    output_type: str
    input_schema: str
    output_schema: str
    description: str
    domain_facts: List[str] = field(default_factory=list)
    sample_inputs: List[Dict[str, Any]] = field(default_factory=list)
    sample_labels: List[Any] = field(default_factory=list)
    known_transforms: List[Dict[str, str]] = field(default_factory=list)
