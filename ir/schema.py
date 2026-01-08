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

    # 输入变换
    transform: Callable  # 输入变换函数：kwargs_dict -> modified_kwargs_dict
    transform_code: str = ""  # 输入变换代码（lambda表达式字符串）

    # 输出验证：框架无关的表达式
    # 表达式中的可用变量：
    # - orig: 原始输出
    # - trans: 变换后输出
    # - x: 原始输入（如果是单输入）
    # - tolerance: 容差值
    #
    # 示例表达式：
    # - "orig == trans"  # 相等关系
    # - "trans == 2 * orig"  # 比例关系（缩放因子2）
    # - "trans == -orig"  # 取反关系
    # - "orig + trans == abs(x)"  # 组合关系
    # - "orig == trans"  # 幂等性（f(f(x)) == f(x)）
    oracle_expr: str = ""  # 框架无关的验证表达式

    # 元数据
    category: str = "general"
    # MR类别：linearity, monotonicity, idempotency, composition, invariance, symmetry, boundary
    tolerance: float = 1e-6  # 数值容差
    analysis: str = ""  # 分析说明
    layer: str = "operator"  # MR所属层次
    verified: bool = False  # 是否已通过验证
