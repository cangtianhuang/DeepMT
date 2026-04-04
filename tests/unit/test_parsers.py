"""解析器单元测试：ASTParser + SympyTranslator"""

import pytest
import sympy as sp

from mr_generator.operator.ast_parser import ASTParser
from mr_generator.operator.sympy_translator import SympyTranslator


# ---------------------------------------------------------------------------
# ASTParser
# ---------------------------------------------------------------------------

class TestASTParser:

    def setup_method(self):
        self.parser = ASTParser()

    def test_simple_addition(self):
        code = "def add(x, y):\n    return x + y\n"
        result = self.parser.parse_to_sympy(code)
        x0, x1 = sp.symbols("x0 x1")
        assert sp.simplify(result - (x0 + x1)) == 0

    def test_simple_multiplication(self):
        code = "def mul(x, y):\n    return x * y\n"
        result = self.parser.parse_to_sympy(code)
        x0, x1 = sp.symbols("x0 x1")
        assert sp.simplify(result - x0 * x1) == 0

    def test_abs_function(self):
        code = "def absolute(x):\n    return abs(x)\n"
        result = self.parser.parse_to_sympy(code)
        x0 = sp.Symbol("x0")
        assert result == sp.Abs(x0)

    def test_max_function(self):
        code = "def maximum(x, y):\n    return max(x, y)\n"
        result = self.parser.parse_to_sympy(code)
        x0, x1 = sp.symbols("x0 x1")
        assert result == sp.Max(x0, x1)

    def test_relu_conditional(self):
        code = "def relu(x):\n    return x if x > 0 else 0\n"
        result = self.parser.parse_to_sympy(code)
        assert isinstance(result, sp.Piecewise)

    def test_polynomial(self):
        code = "def poly(x):\n    return x**2 + 2*x + 1\n"
        result = self.parser.parse_to_sympy(code)
        x0 = sp.Symbol("x0")
        assert sp.simplify(result - (x0**2 + 2*x0 + 1)) == 0

    def test_invalid_syntax_returns_none(self):
        result = self.parser.parse_to_sympy("def f(x):\n    return x +\n")
        assert result is None


# ---------------------------------------------------------------------------
# SympyTranslator
# ---------------------------------------------------------------------------

class TestSympyTranslator:

    def setup_method(self):
        self.translator = SympyTranslator()

    def test_from_code_add(self):
        code = "def add(x, y):\n    return x + y\n"
        result = self.translator.translate(code=code, use_proxy_path=False)
        x0, x1 = sp.symbols("x0 x1")
        assert sp.simplify(result - (x0 + x1)) == 0

    def test_from_callable(self):
        def double(x):
            return x * 2

        result = self.translator.translate(func=double, use_proxy_path=False)
        x0 = sp.Symbol("x0")
        assert sp.simplify(result - x0 * 2) == 0

    def test_relu_conditional(self):
        code = "def relu(x):\n    return x if x > 0 else 0\n"
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert isinstance(result, sp.Piecewise)

    def test_power_function(self):
        code = "def sq(x):\n    return x ** 2\n"
        result = self.translator.translate(code=code, use_proxy_path=False)
        x0 = sp.Symbol("x0")
        assert sp.simplify(result - x0**2) == 0

    def test_no_input_returns_none(self):
        result = self.translator.translate()
        assert result is None

    def test_constant_function(self):
        code = "def const(x):\n    return 42\n"
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert result == 42
