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
    """标准化MR对象数据结构

    一个 MR 由两部分组成：

    transform_code / transform（输入侧）
        描述"如何将原始输入变换为测试输入"。
        - transform_code: lambda 字符串，格式为 "lambda k: {**k, 'input': ...}"
          （LLM 生成）或 "lambda x, y: (y, x)"（模板/手写）。
        - transform: 由 transform_code eval 得到的可调用对象，运行时直接调用。
        - 用途：① 实际测试时产生变换后输入；② SymPy 证明时对符号做同等变换。

    oracle_expr（输出侧）
        描述"变换后输出 (trans) 与原始输出 (orig) 之间应满足的数学关系"。
        - 格式：框架无关的 Python 表达式字符串。
        - 可用变量：orig（原始输出）、trans（变换后输出）、x（原始输入张量）。
        - 示例：
            "orig == trans"           # 相等（幂等性、交换律等）
            "trans == 2 * orig"       # 线性缩放
            "trans == -orig"          # 取反（反交换律）
            "orig * trans == 1"       # 倒数关系
            "all((trans == orig + 1) | (x < 0))"  # 条件不变性
        - 用途：① SymPy 符号证明；② 数值执行时的运行时断言。
        - 空字符串表示默认检查相等（orig == trans）。
    """

    id: str  # MR唯一标识
    description: str  # MR描述

    # 输入变换（见上方文档）
    transform: Callable  # transform_code eval 后的可调用对象
    transform_code: str = ""  # 输入变换的 lambda 表达式字符串

    # 输出验证（见上方文档）
    oracle_expr: str = ""  # 框架无关的输出关系断言表达式

    # 元数据
    category: str = "general"
    # MR类别：linearity, monotonicity, idempotency, composition, invariance, symmetry, boundary
    tolerance: float = 1e-6  # 数值容差
    analysis: str = ""  # 分析说明
    layer: str = "operator"  # MR所属层次
    verified: bool = False  # 是否已通过验证
