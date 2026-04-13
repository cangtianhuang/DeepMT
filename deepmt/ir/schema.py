from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

# ── 层次与生命周期常量 ─────────────────────────────────────────────────────────
SubjectType = Literal["operator", "model", "application"]
LifecycleState = Literal["pending", "checked", "proven", "retired"]


# ── 统一被测主体基类 ──────────────────────────────────────────────────────────


@dataclass
class TestSubject:
    """三层被测对象的统一抽象基类。

    所有被测对象（算子、模型、应用）共享本基类字段。
    子类通过 subject_type 声明自身层次；通过 metadata 扩展层特有信息。

    Attributes:
        name:         唯一标识名称（如 "torch.add"、"ResNet50"）
        subject_type: 层次类型（"operator" / "model" / "application"）
        framework:    适用框架；None 表示框架无关
        metadata:     层次专属的键值扩展（不影响基类接口）
    """

    name: str
    subject_type: SubjectType = "operator"
    framework: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── 算子层 IR ────────────────────────────────────────────────────────────────


@dataclass
class OperatorIR(TestSubject):
    """算子被测对象描述。

    继承 TestSubject，subject_type 固定为 "operator"。
    """

    subject_type: SubjectType = "operator"
    inputs: Optional[List[Any]] = None
    outputs: Optional[List[Any]] = None
    properties: Optional[Dict[str, Any]] = None
    api_path: str = ""
    api_style: str = "function"
    input_specs: Optional[List[Dict[str, Any]]] = None


# ── 模型层 IR（Phase I 完善）──────────────────────────────────────────────────


@dataclass
class ModelIR(TestSubject):
    """模型被测对象描述。

    继承 TestSubject，subject_type 固定为 "model"。

    Attributes:
        model_type:      模型结构类型，如 "mlp"、"cnn"、"rnn"、"transformer"
        task_type:       任务类型，如 "classification"、"regression"、"embedding"
        input_shape:     单样本输入形状，如 (3, 224, 224)
        output_shape:    单样本输出形状，如 (10,)
        num_classes:     分类任务的类别数；非分类任务为 None
        layers:          层描述列表（由 GraphAnalyzer 填充）
        connections:     层连接关系列表（由 GraphAnalyzer 填充）
        analysis_summary: 结构分析摘要（由 GraphAnalyzer 填充）
        model_instance:  运行时 PyTorch/框架模型对象（不序列化）
    """

    subject_type: SubjectType = "model"
    model_type: str = ""
    task_type: str = ""
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    num_classes: Optional[int] = None
    layers: List[Any] = field(default_factory=list)
    connections: List[Any] = field(default_factory=list)
    analysis_summary: Dict[str, Any] = field(default_factory=dict)
    model_instance: Optional[Any] = field(default=None, repr=False, compare=False)


# ── 应用层 IR（占位，Phase J 完善）──────────────────────────────────────────────


@dataclass
class ApplicationIR(TestSubject):
    """应用被测对象描述（开发中）。

    继承 TestSubject，subject_type 固定为 "application"。
    """

    subject_type: SubjectType = "application"
    purpose: str = ""
    input_format: str = ""
    output_format: str = ""


# ── MR：统一关系表达 ──────────────────────────────────────────────────────────


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

    lifecycle_state（生命周期）
        统一状态流转入口，替代原先分散的 checked/proven/verified 判断：
        - "pending":  已录入，未经任何验证
        - "checked":  数值 precheck 通过（checked=True）
        - "proven":   SymPy 符号证明通过（proven=True），或人工确认
        - "retired":  已废弃，不参与测试
        旧字段 checked / proven / verified 保留用于向后兼容。
    """

    # ── 核心标识 ────────────────────────────────────────────────────────────
    id: str
    description: str

    # ── 所属主体（G 阶段新增）────────────────────────────────────────────────
    subject_name: str = ""          # 关联的被测主体名称（如 "torch.add"）
    subject_type: SubjectType = "operator"  # 关联的被测层次

    # ── 变换定义 ─────────────────────────────────────────────────────────────
    transform_code: str = ""
    oracle_expr: str = ""

    # ── 分类元数据 ────────────────────────────────────────────────────────────
    category: str = "general"
    tolerance: float = 1e-6
    layer: str = "operator"
    source: str = ""  # "llm" | "template" | "manual"
    applicable_frameworks: Optional[List[str]] = None  # None = 通用

    # ── 生命周期（G 阶段新增统一入口）───────────────────────────────────────────
    lifecycle_state: LifecycleState = "pending"

    # ── 验证状态（保留，兼容旧逻辑）──────────────────────────────────────────────
    checked: Optional[bool] = None  # 数值 precheck 通过
    proven: Optional[bool] = None  # SymPy 符号证明通过
    verified: bool = False  # True iff checked=True AND proven=True，或人工确认

    # ── 用户工作区噪音字段（项目库不序列化）──────────────────────────────────
    analysis: str = ""

    # ── 运行时专用（永不序列化）──────────────────────────────────────────────
    transform: Optional[Callable] = field(default=None, repr=False, compare=False)

    def sync_lifecycle(self) -> None:
        """根据 checked / proven / verified 同步 lifecycle_state。

        用于从旧数据迁移或在验证流水线中保持状态一致。
        """
        if self.verified or self.proven:
            self.lifecycle_state = "proven"
        elif self.checked:
            self.lifecycle_state = "checked"
        else:
            self.lifecycle_state = "pending"


# ── Oracle 评估结果 ───────────────────────────────────────────────────────────


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
