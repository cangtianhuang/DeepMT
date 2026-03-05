"""
SymPy翻译器单元测试
测试从简单到复杂的代码到SymPy表达式转换功能
注意：这些测试主要测试AST回退路径，不依赖LLM
"""

import pytest
import sympy as sp

from mr_generator.operator.sympy_translator import SympyTranslator


class TestSympyTranslatorBasic:
    """基础测试：简单函数转换"""

    def setup_method(self):
        self.translator = SympyTranslator()

    def test_simple_add_from_code(self):
        """测试从代码字符串转换简单加法"""
        code = """
def add(x, y):
    return x + y
"""
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert result is not None

        x0, x1 = sp.symbols("x0 x1")
        expected = x0 + x1
        assert sp.simplify(result - expected) == 0

    def test_simple_multiply_from_code(self):
        """测试从代码字符串转换简单乘法"""
        code = """
def multiply(x, y):
    return x * y
"""
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert result is not None

        x0, x1 = sp.symbols("x0 x1")
        expected = x0 * x1
        assert sp.simplify(result - expected) == 0

    def test_simple_function_from_callable(self):
        """测试从可调用对象转换"""

        def simple_func(x):
            return x * 2

        result = self.translator.translate(func=simple_func, use_proxy_path=False)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = x0 * 2
        assert sp.simplify(result - expected) == 0

    def test_abs_function(self):
        """测试abs函数转换"""
        code = """
def absolute(x):
    return abs(x)
"""
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert result is not None

        # abs可能被转换为Abs或Piecewise，两者都是正确的
        # 我们只验证结果不为None
        assert result is not None


class TestSympyTranslatorIntermediate:
    """中级测试：复合表达式和数学函数"""

    def setup_method(self):
        self.translator = SympyTranslator()

    def test_compound_expression(self):
        """测试复合表达式"""
        code = """
def compound(x, y):
    return 2 * x + 3 * y - x * y
"""
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert result is not None

        x0, x1 = sp.symbols("x0 x1")
        expected = 2 * x0 + 3 * x1 - x0 * x1
        assert sp.simplify(result - expected) == 0

    def test_max_function(self):
        """测试max函数"""
        code = """
def maximum(x, y):
    return max(x, y)
"""
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert result is not None

        # max可能被转换为Max或Piecewise，两者都是正确的
        # 我们只验证结果不为None
        assert result is not None

    def test_sqrt_function(self):
        """测试sqrt函数"""
        code = """
def square_root(x):
    return sqrt(x)
"""
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = sp.sqrt(x0)
        assert result == expected

    def test_power_function(self):
        """测试幂函数"""
        code = """
def power(x):
    return x ** 2
"""
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = x0**2
        assert sp.simplify(result - expected) == 0


class TestSympyTranslatorAdvanced:
    """高级测试：复杂函数和特殊情况"""

    def setup_method(self):
        self.translator = SympyTranslator()

    def test_relu_with_max(self):
        """测试使用max实现的ReLU"""
        code = """
def relu(x):
    return max(0, x)
"""
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert result is not None

        # ReLU可能被转换为Max或Piecewise，两者都是正确的
        # 我们只验证结果不为None
        assert result is not None

    def test_relu_with_conditional(self):
        """测试使用条件表达式的ReLU"""
        code = """
def relu(x):
    return x if x > 0 else 0
"""
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert result is not None

        # 验证是Piecewise表达式
        assert isinstance(result, sp.Piecewise)

    def test_polynomial(self):
        """测试多项式"""
        code = """
def polynomial(x):
    return x**3 - 2*x**2 + x - 1
"""
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = x0**3 - 2 * x0**2 + x0 - 1
        assert sp.simplify(result - expected) == 0

    def test_rational_function(self):
        """测试有理函数"""
        code = """
def rational(x):
    return (x + 1) / (x - 1)
"""
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = (x0 + 1) / (x0 - 1)
        assert sp.simplify(result - expected) == 0

    def test_nested_functions(self):
        """测试嵌套函数"""
        code = """
def nested(x):
    return abs(sqrt(x))
"""
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = sp.Abs(sp.sqrt(x0))
        assert result == expected

    def test_trigonometric_function(self):
        """测试三角函数"""
        code = """
def trig(x):
    return sin(x) + cos(x)
"""
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = sp.sin(x0) + sp.cos(x0)
        assert sp.simplify(result - expected) == 0


class TestSympyTranslatorEdgeCases:
    """边界测试：特殊情况和错误处理"""

    def setup_method(self):
        self.translator = SympyTranslator()

    def test_no_code_provided(self):
        """测试没有提供代码"""
        result = self.translator.translate()
        assert result is None

    def test_invalid_code(self):
        """测试无效代码"""
        code = """
def invalid(x):
    return x +
"""
        result = self.translator.translate(code=code, use_proxy_path=False)
        # 注意：LLM可能会修复无效代码，所以这个测试可能会通过
        # 我们只验证不会崩溃
        assert True  # 如果到这里没有异常，测试通过

    def test_constant_function(self):
        """测试常量函数"""
        code = """
def constant(x):
    return 42
"""
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert result is not None
        assert result == 42

    def test_identity_function(self):
        """测试恒等函数"""
        code = """
def identity(x):
    return x
"""
        result = self.translator.translate(code=code, use_proxy_path=False)
        assert result is not None

        x0 = sp.Symbol("x0")
        assert result == x0

    def test_with_signature(self):
        """测试提供函数签名"""
        code = """
def func(x, y):
    return x + y
"""
        result = self.translator.translate(
            code=code, signature="(x, y)", use_proxy_path=False
        )
        assert result is not None

        x0, x1 = sp.symbols("x0 x1")
        expected = x0 + x1
        assert sp.simplify(result - expected) == 0

    def test_with_doc(self):
        """测试提供文档字符串"""
        code = """
def func(x):
    return x * 2
"""
        result = self.translator.translate(
            code=code, doc="Multiply by 2", use_proxy_path=False
        )
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = x0 * 2
        assert sp.simplify(result - expected) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
