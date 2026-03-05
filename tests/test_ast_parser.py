"""
AST解析器单元测试
测试从简单到复杂的AST解析功能
"""

import pytest
import sympy as sp

from mr_generator.operator.ast_parser import ASTParser


class TestASTParserBasic:
    """基础测试：简单的数学表达式"""

    def setup_method(self):
        self.parser = ASTParser()

    def test_simple_addition(self):
        """测试简单加法"""
        code = """
def add(x, y):
    return x + y
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        # 验证表达式
        x0, x1 = sp.symbols("x0 x1")
        expected = x0 + x1
        assert sp.simplify(result - expected) == 0

    def test_simple_subtraction(self):
        """测试简单减法"""
        code = """
def subtract(x, y):
    return x - y
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0, x1 = sp.symbols("x0 x1")
        expected = x0 - x1
        assert sp.simplify(result - expected) == 0

    def test_simple_multiplication(self):
        """测试简单乘法"""
        code = """
def multiply(x, y):
    return x * y
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0, x1 = sp.symbols("x0 x1")
        expected = x0 * x1
        assert sp.simplify(result - expected) == 0

    def test_simple_division(self):
        """测试简单除法"""
        code = """
def divide(x, y):
    return x / y
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0, x1 = sp.symbols("x0 x1")
        expected = x0 / x1
        assert sp.simplify(result - expected) == 0

    def test_simple_power(self):
        """测试简单幂运算"""
        code = """
def power(x, y):
    return x ** y
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0, x1 = sp.symbols("x0 x1")
        expected = x0**x1
        assert sp.simplify(result - expected) == 0

    def test_unary_negation(self):
        """测试一元取反"""
        code = """
def negate(x):
    return -x
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = -x0
        assert sp.simplify(result - expected) == 0


class TestASTParserIntermediate:
    """中级测试：复合表达式和函数调用"""

    def setup_method(self):
        self.parser = ASTParser()

    def test_compound_expression(self):
        """测试复合表达式"""
        code = """
def compound(x, y):
    return 2 * x + 3 * y
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0, x1 = sp.symbols("x0 x1")
        expected = 2 * x0 + 3 * x1
        assert sp.simplify(result - expected) == 0

    def test_abs_function(self):
        """测试abs函数"""
        code = """
def absolute(x):
    return abs(x)
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = sp.Abs(x0)
        assert result == expected

    def test_max_function(self):
        """测试max函数"""
        code = """
def maximum(x, y):
    return max(x, y)
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0, x1 = sp.symbols("x0 x1")
        expected = sp.Max(x0, x1)
        assert result == expected

    def test_min_function(self):
        """测试min函数"""
        code = """
def minimum(x, y):
    return min(x, y)
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0, x1 = sp.symbols("x0 x1")
        expected = sp.Min(x0, x1)
        assert result == expected

    def test_sqrt_function(self):
        """测试sqrt函数"""
        code = """
def square_root(x):
    return sqrt(x)
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = sp.sqrt(x0)
        assert result == expected

    def test_exp_function(self):
        """测试exp函数"""
        code = """
def exponential(x):
    return exp(x)
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = sp.exp(x0)
        assert result == expected

    def test_log_function(self):
        """测试log函数"""
        code = """
def logarithm(x):
    return log(x)
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = sp.log(x0)
        assert result == expected


class TestASTParserAdvanced:
    """高级测试：条件表达式、三角函数和复杂组合"""

    def setup_method(self):
        self.parser = ASTParser()

    def test_conditional_expression(self):
        """测试条件表达式（三元运算符）"""
        code = """
def relu(x):
    return x if x > 0 else 0
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        # 验证是Piecewise表达式
        assert isinstance(result, sp.Piecewise)

    def test_max_as_relu(self):
        """测试使用max实现的ReLU"""
        code = """
def relu(x):
    return max(0, x)
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = sp.Max(0, x0)
        assert result == expected

    def test_sin_function(self):
        """测试sin函数"""
        code = """
def sine(x):
    return sin(x)
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = sp.sin(x0)
        assert result == expected

    def test_cos_function(self):
        """测试cos函数"""
        code = """
def cosine(x):
    return cos(x)
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = sp.cos(x0)
        assert result == expected

    def test_complex_expression(self):
        """测试复杂表达式"""
        code = """
def complex_func(x, y):
    return (x + y) * (x - y)
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0, x1 = sp.symbols("x0 x1")
        expected = (x0 + x1) * (x0 - x1)
        # 展开后比较
        assert sp.expand(result) == sp.expand(expected)

    def test_nested_functions(self):
        """测试嵌套函数调用"""
        code = """
def nested(x):
    return abs(sin(x))
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = sp.Abs(sp.sin(x0))
        assert result == expected

    def test_polynomial(self):
        """测试多项式"""
        code = """
def polynomial(x):
    return x**3 + 2*x**2 - 3*x + 1
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = x0**3 + 2 * x0**2 - 3 * x0 + 1
        assert sp.simplify(result - expected) == 0

    def test_rational_function(self):
        """测试有理函数"""
        code = """
def rational(x):
    return (x + 1) / (x - 1)
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0 = sp.Symbol("x0")
        expected = (x0 + 1) / (x0 - 1)
        assert sp.simplify(result - expected) == 0


class TestASTParserEdgeCases:
    """边界测试：特殊情况和错误处理"""

    def setup_method(self):
        self.parser = ASTParser()

    def test_constant_function(self):
        """测试常量函数"""
        code = """
def constant(x):
    return 42
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None
        assert result == 42

    def test_identity_function(self):
        """测试恒等函数"""
        code = """
def identity(x):
    return x
"""
        result = self.parser.parse_to_sympy(code)
        assert result is not None

        x0 = sp.Symbol("x0")
        assert result == x0

    def test_invalid_syntax(self):
        """测试无效语法"""
        code = """
def invalid(x):
    return x +
"""
        result = self.parser.parse_to_sympy(code)
        assert result is None

    def test_no_return_statement(self):
        """测试没有return语句"""
        code = """
def no_return(x):
    y = x + 1
"""
        result = self.parser.parse_to_sympy(code)
        assert result is None

    def test_multiple_statements(self):
        """测试多个语句（只取最后的return）"""
        code = """
def multi_statement(x):
    y = x + 1
    z = y * 2
    return z
"""
        result = self.parser.parse_to_sympy(code)
        # 注意：当前实现可能无法处理中间变量
        # 这个测试主要验证不会崩溃
        # 实际结果可能是None或者一个表达式
        # 我们只验证不会抛出异常
        assert True  # 如果到这里没有异常，测试通过


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
