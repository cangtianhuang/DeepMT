from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class OperatorIR:
    name: str
    inputs: Optional[List[Any]] = None
    outputs: Optional[List[Any]] = None
    properties: Optional[Dict[str, Any]] = None


@dataclass
class ModelIR:
    name: str
    layers: list
    connections: list


@dataclass
class ApplicationIR:
    name: str
    purpose: str
    input_format: str
    output_format: str


@dataclass
class MetamorphicRelation:
    """标准化MR对象数据结构"""

    id: str  # MR唯一标识
    description: str  # MR描述
    transform: Callable  # 输入变换函数
    expected: str  # 期望关系类型
    tolerance: Optional[float] = None  # 数值容差
    layer: str = "operator"  # MR所属层次
    verified: bool = False  # 是否已通过验证
