from dataclasses import dataclass
from typing import Any, List, Dict, Callable, Optional


@dataclass
class OperatorIR:
    name: str
    inputs: List[Any]
    outputs: List[Any]
    properties: Dict[str, Any]


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
    expected: str  # 期望关系类型: "equal", "proportional", "invariant", etc.
    tolerance: Optional[float] = None  # 数值容差（用于浮点数比较）
    layer: str = "operator"  # MR所属层次: "operator", "model", "application"
    verified: bool = False  # 是否已通过验证（自动推导或SymPy证明）
    
    def __post_init__(self):
        """验证MR对象的有效性"""
        if self.expected not in ["equal", "proportional", "invariant", "monotonic", "custom"]:
            raise ValueError(f"Unsupported expected type: {self.expected}")
