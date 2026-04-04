"""SymPy证明引擎单元测试"""

import uuid
import pytest
import sympy as sp

from deepmt.ir.schema import MetamorphicRelation
from deepmt.mr_generator.operator.sympy_prover import SymPyProver
from deepmt.mr_generator.operator.sympy_translator import SympyTranslator


def make_mr(description, transform, oracle_expr):
    return MetamorphicRelation(
        id=str(uuid.uuid4()),
        description=description,
        transform=transform,
        oracle_expr=oracle_expr,
    )


# ---------------------------------------------------------------------------
# _verify_oracle_expr — 直接传入 SymPy 表达式
# ---------------------------------------------------------------------------

class TestVerifyOracleExpr:

    def setup_method(self):
        self.prover = SymPyProver()
        self.x = sp.Symbol("x0")

    def test_empty_expr_equal(self):
        x = self.x
        proven, _ = self.prover._verify_oracle_expr("", x + 1, x + 1, x)
        assert proven is True

    def test_empty_expr_not_equal(self):
        x = self.x
        proven, _ = self.prover._verify_oracle_expr("", x + 1, x + 2, x)
        assert proven is False

    def test_equality_true(self):
        x = self.x
        proven, _ = self.prover._verify_oracle_expr("orig == trans", x**2, (-x)**2, x)
        assert proven is True

    def test_equality_false(self):
        x = self.x
        proven, _ = self.prover._verify_oracle_expr("orig == trans", x**2, x**3, x)
        assert proven is False

    def test_proportional_expr(self):
        x = self.x
        proven, _ = self.prover._verify_oracle_expr("trans == 4 * orig", x**2, (2*x)**2, x)
        assert proven is True

    def test_composition_relu(self):
        x = self.x
        proven, _ = self.prover._verify_oracle_expr(
            "orig + trans == abs(x)", sp.Max(0, x), sp.Max(0, -x), x
        )
        assert proven is True

    def test_unsupported_format(self):
        x = self.x
        proven, msg = self.prover._verify_oracle_expr("orig + trans", x, x, x)
        assert proven is False
        assert msg is not None

    def test_inequality_not_supported(self):
        x = self.x
        proven, _ = self.prover._verify_oracle_expr("trans >= orig", x**2, (x+1)**2, x)
        assert proven is False


# ---------------------------------------------------------------------------
# prove_mr — 端到端证明
# ---------------------------------------------------------------------------

class TestProveMR:

    def setup_method(self):
        self.prover = SymPyProver(code_translator=SympyTranslator(use_llm=False))

    def test_add_commutative(self):
        mr = make_mr("Commutative: f(x,y)==f(y,x)", lambda x, y: (y, x), "orig == trans")
        proven, err = self.prover.prove_mr(mr=mr, operator_code="def add(x,y):\n    return x+y\n", num_inputs=2)
        assert proven is True
        assert err is None

    def test_abs_symmetry(self):
        mr = make_mr("Symmetry: abs(-x)==abs(x)", lambda x: (-x,), "orig == trans")
        proven, err = self.prover.prove_mr(mr=mr, operator_code="def absolute(x):\n    return abs(x)\n", num_inputs=1)
        assert proven is True

    def test_square_symmetry(self):
        mr = make_mr("Symmetry: sq(-x)==sq(x)", lambda x: (-x,), "orig == trans")
        proven, err = self.prover.prove_mr(mr=mr, operator_code="def square(x):\n    return x**2\n", num_inputs=1)
        assert proven is True

    def test_identity_transform(self):
        mr = make_mr("Identity: f(x)==f(x)", lambda x: (x,), "orig == trans")
        proven, _ = self.prover.prove_mr(mr=mr, operator_code="def sq(x):\n    return x*x\n", num_inputs=1)
        assert proven is True
