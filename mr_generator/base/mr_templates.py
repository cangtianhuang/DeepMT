"""
MR模板池：定义常见数学变换模板
用于路径B（无知识）的MR猜想生成
"""

import uuid
from typing import List, Dict, Callable, Any, Optional
from dataclasses import dataclass

from ir.schema import MetamorphicRelation
from core.logger import get_logger


@dataclass
class MRTemplate:
    """MR模板数据结构"""

    name: str  # 模板名称
    description: str  # MR描述
    transform_func: Callable  # 输入变换函数
    expected: str  # 期望关系类型
    applicable_operators: List[str]  # 适用的算子列表（空列表表示通用）
    min_inputs: int = 1  # 最小输入数量
    max_inputs: Optional[int] = None  # 最大输入数量（None表示无限制）


class MRTemplatePool:
    """MR模板池：管理常见数学变换模板"""

    def __init__(self):
        self.logger = get_logger()
        self.templates: List[MRTemplate] = []
        self._init_templates()

    def _init_templates(self):
        """初始化模板池"""

        # ========== 交换律类 ==========
        self.templates.append(
            MRTemplate(
                name="commutative",
                description="交换律: f(x, y) == f(y, x)",
                transform_func=lambda *args: (
                    (args[1], args[0]) + args[2:] if len(args) >= 2 else args
                ),
                expected="equal",
                applicable_operators=["Add", "Multiply", "MatMul", "Max", "Min"],
                min_inputs=2,
                max_inputs=2,
            )
        )

        # ========== 结合律类 ==========
        self.templates.append(
            MRTemplate(
                name="associative_left",
                description="左结合律: f(f(x, y), z) == f(x, f(y, z))",
                transform_func=lambda x, y, z: (
                    x,
                    (y, z),
                ),  # 简化表示，实际需要特殊处理
                expected="equal",
                applicable_operators=["Add", "Multiply"],
                min_inputs=3,
                max_inputs=3,
            )
        )

        # ========== 单位元类 ==========
        self.templates.append(
            MRTemplate(
                name="additive_identity",
                description="加法单位元: f(x, 0) == x",
                transform_func=lambda x: (x, 0),
                expected="equal",
                applicable_operators=["Add"],
                min_inputs=1,
                max_inputs=1,
            )
        )

        self.templates.append(
            MRTemplate(
                name="multiplicative_identity",
                description="乘法单位元: f(x, 1) == x",
                transform_func=lambda x: (x, 1),
                expected="equal",
                applicable_operators=["Multiply"],
                min_inputs=1,
                max_inputs=1,
            )
        )

        # ========== 反交换律类 ==========
        self.templates.append(
            MRTemplate(
                name="anti_commutative",
                description="反交换律: f(x, y) == -f(y, x)",
                transform_func=lambda *args: (
                    (args[1], args[0]) + args[2:] if len(args) >= 2 else args
                ),
                expected="proportional",  # 需要检查符号相反
                applicable_operators=["Subtract"],
                min_inputs=2,
                max_inputs=2,
            )
        )

        # ========== 倒数关系类 ==========
        self.templates.append(
            MRTemplate(
                name="reciprocal",
                description="倒数关系: f(x, y) == 1 / f(y, x)",
                transform_func=lambda x, y: (y, x),
                expected="proportional",
                applicable_operators=["Divide"],
                min_inputs=2,
                max_inputs=2,
            )
        )

        # ========== 线性变换类 ==========
        self.templates.append(
            MRTemplate(
                name="linear_scaling",
                description="线性缩放: f(k*x, k*y) == k*f(x, y)",
                transform_func=lambda x, y, k=2: (k * x, k * y),
                expected="proportional",
                applicable_operators=["Add", "Multiply", "MatMul"],
                min_inputs=2,
                max_inputs=2,
            )
        )

        self.templates.append(
            MRTemplate(
                name="additive_scaling",
                description="加法缩放: f(x + k, y + k) == f(x, y) + k (对于某些算子)",
                transform_func=lambda x, y, k=1: (x + k, y + k),
                expected="proportional",
                applicable_operators=["Add"],
                min_inputs=2,
                max_inputs=2,
            )
        )

        # ========== 分配律类 ==========
        self.templates.append(
            MRTemplate(
                name="distributive_left",
                description="左分配律: f(x, f(y, z)) == f(f(x, y), f(x, z))",
                transform_func=lambda x, y, z: (x, (y, z)),  # 简化表示
                expected="equal",
                applicable_operators=["Multiply"],  # 乘法对加法的分配律
                min_inputs=3,
                max_inputs=3,
            )
        )

        # ========== 幂等性类 ==========
        self.templates.append(
            MRTemplate(
                name="idempotent",
                description="幂等性: f(x, x) == x",
                transform_func=lambda x: (x, x),
                expected="equal",
                applicable_operators=["Max", "Min"],
                min_inputs=1,
                max_inputs=1,
            )
        )

        # ========== 吸收律类 ==========
        self.templates.append(
            MRTemplate(
                name="absorption_max",
                description="吸收律(Max): f(x, f(x, y)) == f(x, y)",
                transform_func=lambda x, y: (x, (x, y)),  # 简化表示
                expected="equal",
                applicable_operators=["Max"],
                min_inputs=2,
                max_inputs=2,
            )
        )

        self.templates.append(
            MRTemplate(
                name="absorption_min",
                description="吸收律(Min): f(x, f(x, y)) == f(x, y)",
                transform_func=lambda x, y: (x, (x, y)),  # 简化表示
                expected="equal",
                applicable_operators=["Min"],
                min_inputs=2,
                max_inputs=2,
            )
        )

        # ========== 转置关系类 ==========
        self.templates.append(
            MRTemplate(
                name="transpose_matmul",
                description="矩阵乘法转置: f(A, B) == f(B^T, A^T)^T",
                transform_func=lambda A, B: (B.T, A.T),  # 需要特殊处理转置
                expected="equal",
                applicable_operators=["MatMul"],
                min_inputs=2,
                max_inputs=2,
            )
        )

        # ========== 对称性类 ==========
        self.templates.append(
            MRTemplate(
                name="symmetric",
                description="对称性: f(x, y) == f(y, x)",
                transform_func=lambda *args: (
                    (args[1], args[0]) + args[2:] if len(args) >= 2 else args
                ),
                expected="equal",
                applicable_operators=["Dot", "Inner"],
                min_inputs=2,
                max_inputs=2,
            )
        )

        # ========== 平移不变性类 ==========
        self.templates.append(
            MRTemplate(
                name="translation_invariant",
                description="平移不变性: f(x + k, y + k) == f(x, y) + k",
                transform_func=lambda x, y, k=1: (x + k, y + k),
                expected="proportional",
                applicable_operators=["Add"],
                min_inputs=2,
                max_inputs=2,
            )
        )

        # ========== 反转类 ==========
        self.templates.append(
            MRTemplate(
                name="negation",
                description="取反: f(-x, -y) == -f(x, y)",
                transform_func=lambda x, y: (-x, -y),
                expected="proportional",
                applicable_operators=["Add", "Multiply"],
                min_inputs=2,
                max_inputs=2,
            )
        )

        # ========== 零元类 ==========
        self.templates.append(
            MRTemplate(
                name="zero_element_add",
                description="加法零元: f(x, 0) == x",
                transform_func=lambda x: (x, 0),
                expected="equal",
                applicable_operators=["Add"],
                min_inputs=1,
                max_inputs=1,
            )
        )

        self.templates.append(
            MRTemplate(
                name="zero_element_mul",
                description="乘法零元: f(x, 0) == 0",
                transform_func=lambda x: (x, 0),
                expected="equal",
                applicable_operators=["Multiply"],
                min_inputs=1,
                max_inputs=1,
            )
        )

        # ========== 幂运算类 ==========
        self.templates.append(
            MRTemplate(
                name="power_rule",
                description="幂运算规则: f(x^a, x^b) == x^(a+b)",
                transform_func=lambda x, a, b: (x**a, x**b),  # 简化表示
                expected="equal",
                applicable_operators=["Multiply"],  # 对于幂运算
                min_inputs=3,
                max_inputs=3,
            )
        )

        # ========== 对数类 ==========
        self.templates.append(
            MRTemplate(
                name="logarithm_rule",
                description="对数规则: f(log(x), log(y)) == log(x*y)",
                transform_func=lambda x, y: (x, y),  # 需要特殊处理
                expected="equal",
                applicable_operators=["Add"],  # log(x) + log(y) = log(x*y)
                min_inputs=2,
                max_inputs=2,
            )
        )

        # ========== 三角函数类 ==========
        self.templates.append(
            MRTemplate(
                name="trigonometric_identity",
                description="三角恒等式: f(sin(x), cos(x)) == 1",
                transform_func=lambda x: (x, x),  # 需要特殊处理
                expected="equal",
                applicable_operators=["Add"],  # sin^2(x) + cos^2(x) = 1
                min_inputs=1,
                max_inputs=1,
            )
        )

        # ========== 通用模板（适用于所有算子）==========
        self.templates.append(
            MRTemplate(
                name="identity",
                description="恒等变换: f(x) == f(x)",
                transform_func=lambda *args: args,
                expected="equal",
                applicable_operators=[],  # 空列表表示通用
                min_inputs=1,
                max_inputs=None,
            )
        )

        self.logger.info(f"Initialized {len(self.templates)} MR templates")

    def get_applicable_templates(
        self, operator_name: str, num_inputs: int
    ) -> List[MRTemplate]:
        """
        获取适用于指定算子的模板列表

        Args:
            operator_name: 算子名称
            num_inputs: 输入数量

        Returns:
            适用的模板列表
        """
        applicable = []

        for template in self.templates:
            # 检查输入数量
            if num_inputs < template.min_inputs:
                continue
            if template.max_inputs is not None and num_inputs > template.max_inputs:
                continue

            # 检查算子适用性
            if len(template.applicable_operators) == 0:
                # 通用模板
                applicable.append(template)
            elif operator_name in template.applicable_operators:
                applicable.append(template)

        self.logger.debug(
            f"Found {len(applicable)} applicable templates for {operator_name} "
            f"(inputs: {num_inputs})"
        )

        return applicable

    def create_mr_from_template(
        self, template: MRTemplate, operator_inputs: List[Any]
    ) -> MetamorphicRelation:
        """
        从模板创建MR对象

        Args:
            template: MR模板
            operator_inputs: 算子输入

        Returns:
            MetamorphicRelation对象
        """

        # 创建变换函数
        def transform(*args):
            try:
                return template.transform_func(*args)
            except Exception as e:
                self.logger.warning(f"Transform function error: {e}")
                return args

        return MetamorphicRelation(
            id=str(uuid.uuid4()),
            description=template.description,
            transform=transform,
            expected=template.expected,
            tolerance=1e-6,
            layer="operator",
        )

    def generate_mr_candidates(
        self, operator_name: str, operator_inputs: List[Any]
    ) -> List[MetamorphicRelation]:
        """
        为算子生成MR候选列表（路径B：模板池）

        Args:
            operator_name: 算子名称
            operator_inputs: 算子输入

        Returns:
            MR候选列表
        """
        num_inputs = len(operator_inputs)
        templates = self.get_applicable_templates(operator_name, num_inputs)

        candidates = []
        for template in templates:
            try:
                mr = self.create_mr_from_template(template, operator_inputs)
                candidates.append(mr)
            except Exception as e:
                self.logger.warning(
                    f"Failed to create MR from template {template.name}: {e}"
                )

        self.logger.info(
            f"Generated {len(candidates)} MR candidates from templates "
            f"for operator {operator_name}"
        )

        return candidates
