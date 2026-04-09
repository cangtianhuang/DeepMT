from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class OperatorIR:
    name: str
    inputs: Optional[List[Any]] = None
    outputs: Optional[List[Any]] = None
    properties: Optional[Dict[str, Any]] = None
    api_path: str = ""
    api_style: str = "function"
    input_specs: Optional[List[Dict[str, Any]]] = None


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
    """MR 对象数据结构。

    transform_code / transform（输入侧）
        描述"如何将原始输入变换为测试输入"。
        - transform_code: lambda 字符串，格式为 "lambda k: {**k, 'input': ...}"
          （LLM 生成）或 "lambda x, y: (y, x)"（模板/手写）。
        - transform: 由 transform_code eval 得到的可调用对象，运行时直接调用。

    oracle_expr（输出侧）
        描述"变换后输出 (trans) 与原始输出 (orig) 之间应满足的数学关系"。
        - 格式：框架无关的 Python 表达式字符串。
        - 可用变量：orig（原始输出）、trans（变换后输出）、x（原始输入张量）。
        - 示例：
            "orig == trans"
            "trans == 2 * orig"
            "trans == -orig"
    """

    # ── 核心标识 ────────────────────────────────────────────────────────────
    id: str
    description: str

    # ── 变换定义 ─────────────────────────────────────────────────────────────
    transform_code: str = ""
    oracle_expr: str = ""

    # ── 分类元数据 ────────────────────────────────────────────────────────────
    category: str = "general"
    tolerance: float = 1e-6
    layer: str = "operator"
    source: str = ""  # "llm" | "template" | "manual"
    applicable_frameworks: Optional[List[str]] = None  # None = 通用

    # ── 验证状态 ──────────────────────────────────────────────────────────────
    checked: Optional[bool] = None  # 数值 precheck 通过
    proven: Optional[bool] = None  # SymPy 符号证明通过
    verified: bool = False  # True iff checked=True AND proven=True，或人工确认

    # ── 用户工作区噪音字段（项目库不序列化）──────────────────────────────────
    analysis: str = ""

    # ── 运行时专用（永不序列化）──────────────────────────────────────────────
    transform: Optional[Callable] = field(default=None, repr=False, compare=False)


@dataclass
class OracleResult:
    """单次 MR oracle 表达式评估的结果。

    Attributes:
        passed:               MR 是否满足
        expr:                 评估使用的 oracle 表达式字符串
        actual_diff:          实测最大绝对差值 max|lhs - rhs|
        tolerance:            配置的数值容差阈值
        detail:               补充信息（如失败原因、SHAPE_MISMATCH 等）
        max_rel_diff:         最大相对差值；仅等值（==）路径有效
        mismatched_elements:  违规元素数
        total_elements:       参与比较的总元素数
    """

    passed: bool
    expr: str
    actual_diff: float
    tolerance: float
    detail: str = ""
    max_rel_diff: float = float("inf")
    mismatched_elements: int = 0
    total_elements: int = 0
