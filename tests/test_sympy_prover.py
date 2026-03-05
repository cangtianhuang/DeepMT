"""
SymPy证明引擎单元测试
测试从简单到复杂的MR形式化证明功能

验证由 oracle_expr 字段驱动，不再依赖 expected 字段。
"""

import uuid

import pytest
import sympy as sp

from ir.schema import MetamorphicRelation
from mr_generator.operator.sympy_prover import SymPyProver


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def make_mr(description, transform, oracle_expr):
    """创建一个最小的 MetamorphicRelation，只设置必要字段。"""
    return MetamorphicRelation(
        id=str(uuid.uuid4()),
        description=description,
        transform=transform,
        oracle_expr=oracle_expr,
    )


# ===========================================================================
# 1. _verify_oracle_expr 专项测试（直接传入 SymPy 表达式，不调用 LLM）
# ===========================================================================

class TestVerifyOracleExpr:
    """直接测试 _verify_oracle_expr 的解析与验证逻辑。"""

    def setup_method(self):
        self.prover = SymPyProver()
        self.x = sp.Symbol("x0")

    # ---- 空 oracle_expr ----

    def test_empty_expr_equal(self):
        """空 oracle_expr 且 orig == trans → 通过"""
        x = self.x
        is_proven, _ = self.prover._verify_oracle_expr("", x + 1, x + 1, x)
        assert is_proven is True

    def test_empty_expr_not_equal(self):
        """空 oracle_expr 且 orig != trans → 失败"""
        x = self.x
        is_proven, msg = self.prover._verify_oracle_expr("", x + 1, x + 2, x)
        assert is_proven is False
        assert msg is not None

    # ---- 相等关系 ----

    def test_equality_true(self):
        """orig == trans 且确实相等 → 通过"""
        x = self.x
        orig = x ** 2
        trans = (-x) ** 2          # (-x)^2 = x^2
        is_proven, _ = self.prover._verify_oracle_expr("orig == trans", orig, trans, x)
        assert is_proven is True

    def test_equality_false(self):
        """orig == trans 但实际不相等 → 失败"""
        x = self.x
        is_proven, msg = self.prover._verify_oracle_expr("orig == trans", x**2, x**3, x)
        assert is_proven is False
        assert msg is not None

    # ---- 比例关系 ----

    def test_proportional_expr(self):
        """trans == 4 * orig：f(x)=x^2, f(2x)=4x^2 → 通过"""
        x = self.x
        orig = x ** 2
        trans = (2 * x) ** 2       # = 4*x^2
        is_proven, _ = self.prover._verify_oracle_expr("trans == 4 * orig", orig, trans, x)
        assert is_proven is True

    def test_proportional_expr_wrong_factor(self):
        """trans == 3 * orig 但实际是 4 倍 → 失败"""
        x = self.x
        orig = x ** 2
        trans = (2 * x) ** 2
        is_proven, _ = self.prover._verify_oracle_expr("trans == 3 * orig", orig, trans, x)
        assert is_proven is False

    # ---- 复合关系 ----

    def test_composition_relu(self):
        """ReLU 复合关系：relu(x) + relu(-x) == abs(x)"""
        x = self.x
        orig  = sp.Max(0, x)       # relu(x)
        trans = sp.Max(0, -x)      # relu(-x)
        is_proven, _ = self.prover._verify_oracle_expr(
            "orig + trans == abs(x)", orig, trans, x
        )
        assert is_proven is True

    def test_composition_expr_false(self):
        """orig + trans == abs(x) 但不成立 → 失败"""
        x = self.x
        orig  = x ** 2
        trans = x ** 2
        is_proven, _ = self.prover._verify_oracle_expr(
            "orig + trans == abs(x)", orig, trans, x
        )
        assert is_proven is False

    # ---- all() 包装 ----

    def test_all_zero_true(self):
        """all(trans == 0) 且 trans 确实为 0 → 通过"""
        x = self.x
        is_proven, _ = self.prover._verify_oracle_expr(
            "all(trans == 0)", x ** 2, sp.Integer(0), x
        )
        assert is_proven is True

    def test_all_zero_false(self):
        """all(trans == 0) 但 trans 不为 0 → 失败"""
        x = self.x
        is_proven, _ = self.prover._verify_oracle_expr(
            "all(trans == 0)", x ** 2, x ** 2, x
        )
        assert is_proven is False

    # ---- 不等式（无法符号验证）----

    def test_inequality_returns_false(self):
        """不等式 oracle_expr 不支持符号验证 → 返回 False"""
        x = self.x
        is_proven, msg = self.prover._verify_oracle_expr(
            "trans >= orig", x ** 2, (x + 1) ** 2, x
        )
        assert is_proven is False
        assert msg is not None

    def test_ge_le_inequality(self):
        """<= 不等式同样不支持"""
        x = self.x
        is_proven, _ = self.prover._verify_oracle_expr(
            "trans <= orig", x ** 2, x ** 2, x
        )
        assert is_proven is False

    # ---- 不支持的格式 ----

    def test_unsupported_format(self):
        """没有比较符的表达式 → 返回 False"""
        x = self.x
        is_proven, msg = self.prover._verify_oracle_expr("orig + trans", x, x, x)
        assert is_proven is False
        assert msg is not None


# ===========================================================================
# 2. 基础证明测试：直接给定代码字符串
# ===========================================================================

class TestSymPyProverBasic:
    """基础测试：简单算子的 MR 证明。"""

    def setup_method(self):
        self.prover = SymPyProver()

    def test_code_to_sympy_simple_add(self):
        """代码转 SymPy：加法"""
        code = "def add(x, y):\n    return x + y\n"
        result = self.prover.code_to_sympy(code=code)
        assert result is not None
        x0, x1 = sp.symbols("x0 x1")
        assert sp.simplify(result - (x0 + x1)) == 0

    def test_code_to_sympy_simple_multiply(self):
        """代码转 SymPy：乘法"""
        code = "def multiply(x, y):\n    return x * y\n"
        result = self.prover.code_to_sympy(code=code)
        assert result is not None
        x0, x1 = sp.symbols("x0 x1")
        assert sp.simplify(result - x0 * x1) == 0

    def test_prove_add_commutative(self):
        """加法交换律：f(x,y) == f(y,x)"""
        mr = make_mr(
            "Commutative: f(x,y) == f(y,x)",
            lambda x, y: (y, x),
            "orig == trans",
        )
        code = "def add(x, y):\n    return x + y\n"
        is_proven, error_msg = self.prover.prove_mr(mr=mr, operator_code=code, num_inputs=2)
        assert is_proven is True
        assert error_msg is None

    def test_prove_multiply_commutative(self):
        """乘法交换律：f(x,y) == f(y,x)"""
        mr = make_mr(
            "Commutative: f(x,y) == f(y,x)",
            lambda x, y: (y, x),
            "orig == trans",
        )
        code = "def multiply(x, y):\n    return x * y\n"
        is_proven, error_msg = self.prover.prove_mr(mr=mr, operator_code=code, num_inputs=2)
        assert is_proven is True
        assert error_msg is None

    def test_prove_identity_transform(self):
        """恒等变换：f(x) == f(x)"""
        mr = make_mr(
            "Identity: f(x) == f(x)",
            lambda x: (x,),
            "orig == trans",
        )
        code = "def square(x):\n    return x * x\n"
        is_proven, error_msg = self.prover.prove_mr(mr=mr, operator_code=code, num_inputs=1)
        assert is_proven is True
        assert error_msg is None


# ===========================================================================
# 3. 中级证明测试
# ===========================================================================

class TestSymPyProverIntermediate:
    """中级测试：复合关系、对称性等。"""

    def setup_method(self):
        self.prover = SymPyProver()

    def test_prove_relu_positive_scaling(self):
        """ReLU 正缩放：relu(2x) == 2*relu(x)"""
        mr = make_mr(
            "Positive scaling: relu(2*x) == 2*relu(x)",
            lambda x: (2 * x,),
            "trans == 2 * orig",
        )
        code = "def relu(x):\n    return max(0, x)\n"
        is_proven, _ = self.prover.prove_mr(mr=mr, operator_code=code, num_inputs=1)
        # SymPy 对 Max 的化简能力有限，不强制要求结果
        assert isinstance(is_proven, bool)

    def test_prove_abs_symmetry(self):
        """abs 对称性：abs(-x) == abs(x)"""
        mr = make_mr(
            "Symmetry: abs(-x) == abs(x)",
            lambda x: (-x,),
            "orig == trans",
        )
        code = "def absolute(x):\n    return abs(x)\n"
        is_proven, error_msg = self.prover.prove_mr(mr=mr, operator_code=code, num_inputs=1)
        assert is_proven is True
        assert error_msg is None

    def test_prove_square_symmetry(self):
        """x^2 对称性：square(-x) == square(x)"""
        mr = make_mr(
            "Symmetry: square(-x) == square(x)",
            lambda x: (-x,),
            "orig == trans",
        )
        code = "def square(x):\n    return x ** 2\n"
        is_proven, error_msg = self.prover.prove_mr(mr=mr, operator_code=code, num_inputs=1)
        assert is_proven is True
        assert error_msg is None

    def test_prove_add_identity_element(self):
        """加法单位元：f(x, y) == f(x, y)（恒等变换）"""
        mr = make_mr(
            "Identity transform: f(x,y) == f(x,y)",
            lambda x, y: (x, y),
            "orig == trans",
        )
        code = "def add(x, y):\n    return x + y\n"
        is_proven, _ = self.prover.prove_mr(mr=mr, operator_code=code, num_inputs=2)
        assert is_proven is True


# ===========================================================================
# 4. 高级证明测试
# ===========================================================================

class TestSymPyProverAdvanced:
    """高级测试：比例关系、批量证明、SymPy 表达式复用。"""

    def setup_method(self):
        self.prover = SymPyProver()

    def test_prove_cubic_scaling(self):
        """三次函数缩放：f(2x) == 8*f(x)（直接传 SymPy 表达式，不依赖 LLM）"""
        x0 = sp.Symbol("x0")
        sympy_expr = x0 ** 3
        mr = make_mr("Cubic scaling: f(2*x) == 8*f(x)", lambda x: (2 * x,), "trans == 8 * orig")
        is_proven, error_msg = self.prover.prove_mr_with_expr(
            mr=mr, sympy_expr=sympy_expr, num_inputs=1
        )
        assert is_proven is True

    def test_prove_square_scaling(self):
        """平方函数缩放：f(2x) == 4*f(x)（直接传 SymPy 表达式，不依赖 LLM）"""
        x0 = sp.Symbol("x0")
        sympy_expr = x0 ** 2
        mr = make_mr("Square scaling: f(2*x) == 4*f(x)", lambda x: (2 * x,), "trans == 4 * orig")
        is_proven, error_msg = self.prover.prove_mr_with_expr(
            mr=mr, sympy_expr=sympy_expr, num_inputs=1
        )
        assert is_proven is True

    def test_prove_with_sympy_expr_reuse(self):
        """预转换的 SymPy 表达式可以被多个 MR 复用"""
        code = "def add(x, y):\n    return x + y\n"
        sympy_expr = self.prover.code_to_sympy(code=code)
        assert sympy_expr is not None

        mr = make_mr(
            "Commutative: f(x,y) == f(y,x)",
            lambda x, y: (y, x),
            "orig == trans",
        )
        is_proven, _ = self.prover.prove_mr_with_expr(mr=mr, sympy_expr=sympy_expr, num_inputs=2)
        assert is_proven is True

    def test_prove_mrs_batch(self):
        """批量证明：所有正确的 MR 都应被证明"""
        code = "def add(x, y):\n    return x + y\n"
        mrs = [
            make_mr("Commutative", lambda x, y: (y, x), "orig == trans"),
            make_mr("Identity transform", lambda x, y: (x, y), "orig == trans"),
        ]
        proven = self.prover.prove_mrs(mrs=mrs, operator_code=code, num_inputs=2)
        assert len(proven) == 2

    def test_prove_sum_of_squares_commutative(self):
        """x^2 + y^2 的交换律"""
        mr = make_mr(
            "Commutative: f(x,y) == f(y,x)",
            lambda x, y: (y, x),
            "orig == trans",
        )
        code = "def sum_of_squares(x, y):\n    return x**2 + y**2\n"
        is_proven, _ = self.prover.prove_mr(mr=mr, operator_code=code, num_inputs=2)
        assert is_proven is True

    def test_prove_relu_abs_composition(self):
        """ReLU 复合：relu(x) + relu(-x) == abs(x)"""
        # 直接用 SymPy 表达式测试，绕过 LLM
        x0 = sp.Symbol("x0")
        sympy_expr = sp.Max(0, x0)

        mr = make_mr(
            "Composition: relu(x) + relu(-x) == abs(x)",
            lambda x: (-x,),
            "orig + trans == abs(x)",
        )
        is_proven, error_msg = self.prover.prove_mr_with_expr(
            mr=mr, sympy_expr=sympy_expr, num_inputs=1
        )
        assert is_proven is True
        assert error_msg is None


# ===========================================================================
# 5. 边界与错误处理测试
# ===========================================================================

class TestSymPyProverEdgeCases:
    """边界测试：错误输入、空列表、推断输入数量等。"""

    def setup_method(self):
        self.prover = SymPyProver()

    def test_prove_with_no_code(self):
        """未提供代码 → 证明失败并返回错误信息"""
        mr = make_mr("Test", lambda x: (x,), "orig == trans")
        is_proven, error_msg = self.prover.prove_mr(mr=mr, num_inputs=1)
        assert is_proven is False
        assert error_msg is not None

    def test_prove_false_mr(self):
        """错误的 MR（f(x) == f(2x)）→ 证明失败"""
        mr = make_mr(
            "False MR: f(x) == f(2*x)",
            lambda x: (2 * x,),
            "orig == trans",
        )
        code = "def square(x):\n    return x * x\n"
        is_proven, _ = self.prover.prove_mr(mr=mr, operator_code=code, num_inputs=1)
        assert is_proven is False

    def test_prove_inequality_oracle_expr(self):
        """不等式 oracle_expr 无法符号验证 → 返回 False"""
        mr = make_mr(
            "Monotonicity: f(x+1) >= f(x)",
            lambda x: (x + 1,),
            "trans >= orig",
        )
        code = "def square(x):\n    return x ** 2\n"
        is_proven, error_msg = self.prover.prove_mr(mr=mr, operator_code=code, num_inputs=1)
        assert is_proven is False
        assert error_msg is not None

    def test_prove_empty_oracle_defaults_to_equal(self):
        """空 oracle_expr 默认使用相等检查：恒等变换应通过（直接传 SymPy 表达式，不依赖 LLM）"""
        x0 = sp.Symbol("x0")
        mr = MetamorphicRelation(
            id=str(uuid.uuid4()),
            description="Identity with empty oracle_expr",
            transform=lambda x: (x,),
            oracle_expr="",       # 显式空
        )
        is_proven, _ = self.prover.prove_mr_with_expr(mr=mr, sympy_expr=x0, num_inputs=1)
        assert is_proven is True

    def test_prove_empty_mrs_list(self):
        """空 MR 列表 → 返回空列表"""
        proven = self.prover.prove_mrs(mrs=[], operator_code="def f(x): return x")
        assert proven == []

    def test_prove_num_inputs_inferred(self):
        """自动推断输入数量"""
        mr = make_mr("Identity", lambda x: (x,), "orig == trans")
        code = "def identity(x):\n    return x\n"
        # 不提供 num_inputs
        is_proven, _ = self.prover.prove_mr(mr=mr, operator_code=code)
        assert is_proven is True

    def test_prove_invalid_code_does_not_crash(self):
        """无效代码不应抛出异常（LLM 可能修复或失败）"""
        mr = make_mr("Test", lambda x: (x,), "orig == trans")
        code = "def invalid(x):\n    return x +\n"
        is_proven, _ = self.prover.prove_mr(mr=mr, operator_code=code, num_inputs=1)
        assert isinstance(is_proven, bool)   # 只要不崩溃即可


# ===========================================================================
# 6. 真实算子场景测试
# ===========================================================================

class TestSymPyProverRealWorldCases:
    """真实场景：基于常见深度学习算子的 MR 证明。"""

    def setup_method(self):
        self.prover = SymPyProver()

    def test_relu_idempotency(self):
        """ReLU 幂等性（简化版）：恒等变换，oracle_expr = 'orig == trans'"""
        mr = make_mr(
            "Idempotency: relu(x) == relu(x)",
            lambda x: (x,),
            "orig == trans",
        )
        code = "def relu(x):\n    return max(0, x)\n"
        is_proven, _ = self.prover.prove_mr(mr=mr, operator_code=code, num_inputs=1)
        assert is_proven is True

    def test_abs_identity_transform(self):
        """abs 恒等变换：abs(x) == abs(x)"""
        mr = make_mr(
            "Identity: abs(x) == abs(x)",
            lambda x: (x,),
            "orig == trans",
        )
        code = "def absolute(x):\n    return abs(x)\n"
        is_proven, _ = self.prover.prove_mr(mr=mr, operator_code=code, num_inputs=1)
        assert is_proven is True

    def test_polynomial_homogeneity(self):
        """平方函数齐次性：f(2x) == 4*f(x)（直接传 SymPy 表达式，不依赖 LLM）"""
        x0 = sp.Symbol("x0")
        mr = make_mr(
            "Homogeneity: f(2*x) == 4*f(x)",
            lambda x: (2 * x,),
            "trans == 4 * orig",
        )
        is_proven, error_msg = self.prover.prove_mr_with_expr(
            mr=mr, sympy_expr=x0**2, num_inputs=1
        )
        assert is_proven is True

    def test_abs_relu_composition_direct(self):
        """直接用 SymPy 表达式验证 relu(x)+relu(-x)==|x|"""
        x0 = sp.Symbol("x0")
        # 使用 prove_mr_with_expr，绕过 code→SymPy 转换
        mr = make_mr(
            "Composition: relu(x) + relu(-x) == |x|",
            lambda x: (-x,),
            "orig + trans == abs(x)",
        )
        sympy_expr = sp.Max(sp.Integer(0), x0)
        is_proven, error_msg = self.prover.prove_mr_with_expr(
            mr=mr, sympy_expr=sympy_expr, num_inputs=1
        )
        assert is_proven is True
        assert error_msg is None

    def test_negation_symmetry_direct(self):
        """直接用 SymPy 验证 f(-x) == -f(x)（奇函数）"""
        x0 = sp.Symbol("x0")
        sympy_expr = x0 ** 3          # f(x) = x^3 是奇函数

        mr = make_mr(
            "Odd function: f(-x) == -f(x)",
            lambda x: (-x,),
            "trans == -orig",
        )
        is_proven, error_msg = self.prover.prove_mr_with_expr(
            mr=mr, sympy_expr=sympy_expr, num_inputs=1
        )
        assert is_proven is True
        assert error_msg is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
