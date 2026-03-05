"""
SymPy证明引擎：使用SymPy进行形式化证明
使用 code_to_sympy 动态将 Python 代码转化为 sympy 表达式
然后根据 oracle_expr 进行形式化验证
"""

from typing import Callable, List, Optional, Tuple

import sympy as sp

from core.logger import get_logger, log_structured
from ir.schema import MetamorphicRelation
from mr_generator.operator.sympy_translator import SympyTranslator


class _MockTorch:
    """SymPy兼容的 torch mock，用于在符号变换中替代 torch 函数。

    当 LLM 生成的 transform_code 包含 torch.relu / torch.abs 等调用时，
    可用此 mock 在 SymPy 上下文中重新 eval 该 lambda，从而进行符号推导。
    """

    relu = staticmethod(lambda x: sp.Max(0, x))
    relu_ = staticmethod(lambda x: sp.Max(0, x))
    abs = staticmethod(lambda x: sp.Abs(x))
    zeros_like = staticmethod(lambda x: sp.Integer(0))
    ones_like = staticmethod(lambda x: sp.Integer(1))
    exp = staticmethod(lambda x: sp.exp(x))
    log = staticmethod(lambda x: sp.log(x))
    tanh = staticmethod(lambda x: sp.tanh(x))
    sqrt = staticmethod(lambda x: sp.sqrt(x))
    sigmoid = staticmethod(lambda x: sp.Rational(1) / (1 + sp.exp(-x)))
    square = staticmethod(lambda x: x**2)
    neg = staticmethod(lambda x: -x)
    sign = staticmethod(lambda x: sp.sign(x))

    @staticmethod
    def clamp(x, min=None, max=None):
        result = x
        if min is not None:
            result = sp.Max(min, result)
        if max is not None:
            result = sp.Min(max, result)
        return result

    @staticmethod
    def pow(x, exponent):
        return x**exponent

    def __getattr__(self, name):
        return lambda *args, **kwargs: args[0] if args else sp.Integer(0)


_MOCK_TORCH = _MockTorch()

# oracle_expr 关键字映射（向后兼容旧模板的 expected 字段）
_ORACLE_KEYWORD_MAP = {
    "equal": "",      # 空字符串 → 默认相等检查
    "equiv": "",
}


class SymPyProver:
    """SymPy形式化证明引擎：使用动态代码转换 + oracle_expr 验证"""

    def __init__(self, code_translator: Optional[SympyTranslator] = None):
        """
        初始化SymPy证明引擎

        Args:
            code_translator: 代码转换器（如果为None则创建默认实例）
        """
        self.logger = get_logger(self.__class__.__name__)
        self.code_translator = code_translator or SympyTranslator()

    # ------------------------------------------------------------------
    # 代码转 SymPy 表达式
    # ------------------------------------------------------------------

    def code_to_sympy(
        self,
        code: Optional[str] = None,
        func: Optional[Callable] = None,
        doc: Optional[str] = None,
        signature: Optional[str] = None,
        num_inputs: Optional[int] = None,
    ) -> Optional[sp.Expr]:
        """
        动态将 Python 代码转化为 sympy 表达式

        Args:
            code: Python代码字符串
            func: Python函数对象
            doc: 函数文档字符串
            signature: 函数签名字符串
            num_inputs: 输入数量（保留参数，不传递给translator）

        Returns:
            SymPy表达式，如果转换失败则返回None
        """
        sympy_expr = self.code_translator.translate(
            code=code,
            func=func,
            doc=doc,
            signature=signature,
        )
        if sympy_expr is None:
            self.logger.warning("Failed to convert code to SymPy expression")
        return sympy_expr

    # ------------------------------------------------------------------
    # oracle_expr 解析与验证
    # ------------------------------------------------------------------

    def _verify_oracle_expr(
        self,
        oracle_expr: str,
        orig: sp.Expr,
        trans: sp.Expr,
        x: Optional[sp.Expr] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        解析并验证 oracle_expr

        支持的格式（等式类，可符号验证）：
          - "orig == trans"            相等
          - "trans == 2 * orig"        比例（含任意系数）
          - "orig + trans == abs(x)"   复合
          - "all(trans == 0)"          全零（去掉 all() 包装后等价于 "trans == 0"）
          - ""                         空：默认执行相等检查

        不支持（无法一般性符号验证）：
          - "trans >= orig" 等不等式

        Args:
            oracle_expr: 框架无关的验证表达式字符串
            orig: 原始输出的 SymPy 表达式
            trans: 变换后输出的 SymPy 表达式
            x: 原始输入符号（单输入时用于 oracle_expr 中的 x 变量）

        Returns:
            (是否验证成功, 错误信息)
        """
        # 空表达式或关键字 "equal"：默认检查相等性
        oracle_resolved = oracle_expr.strip() if oracle_expr else ""

        # 向后兼容：处理旧模板使用关键字而非等式字符串的情况
        if oracle_resolved in _ORACLE_KEYWORD_MAP:
            oracle_resolved = _ORACLE_KEYWORD_MAP[oracle_resolved]

        if not oracle_resolved:
            return self._check_equality(orig, trans)

        try:
            oracle_clean = oracle_resolved

            # 去掉 all(...) 包装（SymPy 对标量表达式不需要 all()）
            if oracle_clean.startswith("all(") and oracle_clean.endswith(")"):
                oracle_clean = oracle_clean[4:-1]

            # 构建求值上下文：Python 内置函数映射到对应的 SymPy 版本
            x_sym = x if x is not None else sp.Symbol("x")
            ctx = {
                "orig": orig,
                "trans": trans,
                "x": x_sym,
                "abs": sp.Abs,
                "max": sp.Max,
                "min": sp.Min,
                "sqrt": sp.sqrt,
                "exp": sp.exp,
                "log": sp.log,
                "sp": sp,
            }

            # 临时替换 != 避免误匹配 ==
            temp = oracle_clean.replace("!=", "\x00NEQ\x00")

            if "==" in temp:
                # 取第一个 == 作为等号位置
                eq_idx = temp.index("==")
                lhs_str = temp[:eq_idx].strip().replace("\x00NEQ\x00", "!=")
                rhs_str = temp[eq_idx + 2 :].strip().replace("\x00NEQ\x00", "!=")

                lhs = eval(lhs_str, {"__builtins__": {}}, ctx)  # noqa: S307
                rhs = eval(rhs_str, {"__builtins__": {}}, ctx)  # noqa: S307

                ok, msg = self._sympy_equals(lhs, rhs, lhs_str, rhs_str)
                return ok, msg

            elif any(op in oracle_clean for op in [">=", "<=", ">", "<"]):
                # 不等式在没有域约束的情况下无法一般性符号验证
                return (
                    False,
                    f"Cannot symbolically verify inequality oracle_expr: {oracle_expr}",
                )

            else:
                return False, f"Unsupported oracle_expr format: {oracle_expr}"

        except Exception as e:
            self.logger.debug(f"Error verifying oracle_expr '{oracle_expr}': {e}")
            return False, f"Error verifying oracle_expr '{oracle_expr}': {e}"

    def _check_equality(
        self, orig: sp.Expr, trans: sp.Expr
    ) -> Tuple[bool, Optional[str]]:
        """检查 orig == trans（多策略简化）。"""
        return self._sympy_equals(orig, trans, "orig", "trans")

    def _sympy_equals(
        self,
        lhs: sp.Expr,
        rhs: sp.Expr,
        lhs_str: str = "lhs",
        rhs_str: str = "rhs",
    ) -> Tuple[bool, Optional[str]]:
        """
        用多种策略判断 lhs == rhs（即 lhs - rhs == 0）。

        策略顺序：
          1. sp.simplify(diff)
          2. diff.rewrite(Abs) 后再 simplify（处理 Max/Min）
          3. piecewise_fold 后再 simplify（处理嵌套 Piecewise）
          4. 数值采样（对无法符号简化的 Piecewise 等复杂表达式）
        """
        try:
            diff = sp.simplify(lhs - rhs)
        except Exception:
            diff = lhs - rhs

        if diff == 0:
            return True, None

        # 策略2：rewrite Max/Min 为 Abs
        try:
            diff2 = sp.simplify(diff.rewrite(sp.Abs))
            if diff2 == 0:
                return True, None
        except Exception:
            diff2 = diff

        # 策略3：Piecewise 展开后再化简
        try:
            diff3 = sp.simplify(sp.piecewise_fold(diff))
            if diff3 == 0:
                return True, None
            diff3b = sp.simplify(diff3.rewrite(sp.Abs))
            if diff3b == 0:
                return True, None
        except Exception:
            diff3 = diff

        # 策略4：数值采样（对含 Piecewise/Max/Min 的复杂表达式）
        if self._check_numerically_zero(diff):
            return True, None

        return (
            False,
            f"Oracle proof failed: simplify(({lhs_str}) - ({rhs_str})) = {diff} ≠ 0",
        )

    def _check_numerically_zero(
        self, expr: sp.Expr, tolerance: float = 1e-9
    ) -> bool:
        """
        通过多点数值采样判断表达式是否恒为零。

        对无法符号简化的 Piecewise/Max 表达式，使用数值验证作为最终判断。
        采样范围：-10 ~ 10，正负各取若干点，覆盖各分支条件。
        """
        free_syms = list(expr.free_symbols)
        if not free_syms:
            try:
                return abs(complex(expr.evalf())) < tolerance
            except Exception:
                return False

        # 覆盖正、负、零三个区域
        sample_values = [-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0, 10.0]
        for val in sample_values:
            point = {sym: val for sym in free_syms}
            try:
                result = expr.subs(point).evalf()
                if abs(complex(result)) > tolerance:
                    return False
            except Exception:
                return False
        return True

    # ------------------------------------------------------------------
    # 符号变换应用（支持位置参数式和字典式两种 transform 格式）
    # ------------------------------------------------------------------

    def _apply_transform_symbolically(
        self,
        mr: MetamorphicRelation,
        symbols: list,
    ) -> Optional[tuple]:
        """
        将 MR 的 transform 应用于 SymPy 符号，返回变换后的符号元组。

        支持两种 transform 格式：
          1. 位置参数式（模板/手写）：lambda x, y: (y, x)
             直接调用 mr.transform(*symbols)
          2. 字典式（LLM生成）：lambda k: {**k, 'input': 2.0 * k['input']}
             使用 mock torch 重新 eval transform_code，再以符号字典调用

        Returns:
            变换后的符号元组，失败时返回 None
        """
        # --- 路径 1：位置参数式 transform ---
        try:
            result = mr.transform(*symbols)
            if not isinstance(result, (tuple, list)):
                result = (result,)
            # 确保每个元素都是 SymPy 表达式
            return tuple(
                sp.sympify(r) if not isinstance(r, sp.Basic) else r for r in result
            )
        except (TypeError, AttributeError, KeyError):
            pass  # 可能是字典式 transform，继续尝试
        except Exception as e:
            self.logger.debug(f"Position-based transform failed: {e}")

        # --- 路径 2：字典式 transform（LLM生成，需 mock torch）---
        transform_code = getattr(mr, "transform_code", "") or ""
        transform_code = transform_code.strip()

        if transform_code.startswith("lambda"):
            try:
                # 用 mock torch 重新 eval transform_code，使符号运算可用
                eval_ctx = {
                    "__builtins__": {},
                    "torch": _MOCK_TORCH,
                    "sp": sp,
                    "abs": sp.Abs,
                    "max": sp.Max,
                    "min": sp.Min,
                }
                sym_transform = eval(transform_code, eval_ctx)  # noqa: S307

                # 构建代理字典（主输入 → x0，其余用默认值）
                proxy = {"input": symbols[0]}
                if len(symbols) >= 2:
                    proxy["inplace"] = symbols[1]

                result_dict = sym_transform(proxy)

                if isinstance(result_dict, dict) and "input" in result_dict:
                    new_input = result_dict["input"]
                    if not isinstance(new_input, sp.Basic):
                        new_input = sp.sympify(new_input)
                    return (new_input,)

            except Exception as e:
                self.logger.debug(f"Dict-based transform fallback failed: {e}")

        return None

    # ------------------------------------------------------------------
    # 单个 MR 证明
    # ------------------------------------------------------------------

    def prove_mr_with_expr(
        self,
        mr: MetamorphicRelation,
        sympy_expr: sp.Expr,
        num_inputs: int,
        operator_name: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        使用已有的 SymPy 表达式证明 MR（推荐方法，避免重复转换）

        验证逻辑完全由 mr.oracle_expr 决定：
          - "orig == trans"           相等关系
          - "trans == 2 * orig"       比例关系
          - "orig + trans == abs(x)"  复合关系
          - "" (空)                   默认使用相等检查

        Args:
            mr: 蜕变关系
            sympy_expr: 已转换的 SymPy 表达式
            num_inputs: 输入数量
            operator_name: 算子名称（可选，用于日志）

        Returns:
            (是否证明成功, 错误信息)
        """
        try:
            symbols = [sp.Symbol(f"x{i}") for i in range(num_inputs)]

            # 原始输出
            orig = sympy_expr

            # 应用 MR 变换，计算变换后输出
            transformed_symbols = self._apply_transform_symbolically(mr, symbols)
            if transformed_symbols is None:
                error_msg = f"Cannot apply transform symbolically for MR: {mr.description}"
                self.logger.debug(error_msg)
                return False, error_msg

            if not isinstance(transformed_symbols, (tuple, list)):
                transformed_symbols = (transformed_symbols,)

            # 使用 simultaneous=True 确保多变量替换同时进行（避免交换律等证明出错）
            subs_dict = {
                symbols[i]: transformed_symbols[i]
                for i in range(min(len(symbols), len(transformed_symbols)))
            }
            trans = (
                sympy_expr.subs(subs_dict, simultaneous=True) if subs_dict else sympy_expr
            )

            # 原始输入符号（供 oracle_expr 中的 x 变量使用）
            x = symbols[0] if symbols else None

            is_proven, error_msg = self._verify_oracle_expr(
                oracle_expr=mr.oracle_expr,
                orig=orig,
                trans=trans,
                x=x,
            )

            if is_proven:
                self.logger.debug(f"MR proven: {mr.description}")
            else:
                self.logger.debug(
                    f"MR proof failed: {mr.description}. Reason: {error_msg}"
                )
            return is_proven, error_msg

        except Exception as e:
            error_msg = f"SymPy proof error: {str(e)}"
            self.logger.debug(f"SymPy proof exception: {error_msg}")
            return False, error_msg

    def prove_mr(
        self,
        mr: MetamorphicRelation,
        operator_func: Optional[Callable] = None,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
        operator_name: Optional[str] = None,
        num_inputs: Optional[int] = None,
        sympy_expr: Optional[sp.Expr] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        使用SymPy证明MR

        Args:
            mr: 蜕变关系
            operator_func: 算子函数对象
            operator_code: 算子代码字符串
            operator_doc: 算子文档字符串
            operator_name: 算子名称（可选，用于日志）
            num_inputs: 输入数量
            sympy_expr: 已转换的 SymPy 表达式（可选，如果提供则复用）

        Returns:
            (是否证明成功, 错误信息)
        """
        try:
            if sympy_expr is None:
                sympy_expr = self.code_to_sympy(
                    code=operator_code,
                    func=operator_func,
                    doc=operator_doc,
                    num_inputs=num_inputs,
                )
                if sympy_expr is None:
                    error_msg = (
                        f"Cannot convert operator '{operator_name}' to SymPy expression"
                        if operator_name
                        else "Cannot convert operator to SymPy expression"
                    )
                    if operator_name:
                        self.logger.warning(error_msg)
                    return False, error_msg

            if num_inputs is None:
                free_symbols = sympy_expr.free_symbols
                num_inputs = len(free_symbols) if free_symbols else 1

            return self.prove_mr_with_expr(
                mr=mr,
                sympy_expr=sympy_expr,
                num_inputs=num_inputs,
                operator_name=operator_name,
            )

        except Exception as e:
            error_msg = f"SymPy proof error: {str(e)}"
            self.logger.debug(f"SymPy proof exception: {error_msg}")
            return False, error_msg

    # ------------------------------------------------------------------
    # 批量 MR 证明
    # ------------------------------------------------------------------

    def prove_mrs(
        self,
        mrs: List[MetamorphicRelation],
        operator_func: Optional[Callable] = None,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
        operator_name: Optional[str] = None,
        num_inputs: Optional[int] = None,
        sympy_expr: Optional[sp.Expr] = None,
    ) -> List[MetamorphicRelation]:
        """
        证明MR列表，返回经过证明的MR

        优化：只转换一次代码为 SymPy，对所有 MR 复用

        Args:
            mrs: MR列表
            operator_func: 算子函数对象
            operator_code: 算子代码字符串
            operator_doc: 算子文档字符串
            operator_name: 算子名称（可选，用于日志）
            num_inputs: 输入数量
            sympy_expr: 已转换的 SymPy 表达式（可选，如果提供则复用）

        Returns:
            经过证明的MR列表
        """
        if not mrs:
            return []

        proven_mrs = []
        log_structured(self.logger, "GEN", f"Proving {len(mrs)} MRs using SymPy...")

        if sympy_expr is None:
            self.logger.debug(
                "Converting code to SymPy expression (once for all MRs)..."
            )
            sympy_expr = self.code_to_sympy(
                code=operator_code,
                func=operator_func,
                doc=operator_doc,
                num_inputs=num_inputs,
            )
            if sympy_expr is None:
                self.logger.warning(
                    f"Cannot convert operator '{operator_name or 'unknown'}' to SymPy. "
                    f"All {len(mrs)} MRs will fail proof."
                )
                return []
        else:
            self.logger.debug(
                "Using provided SymPy expression (reusing from previous stage)"
            )

        if num_inputs is None:
            free_symbols = sympy_expr.free_symbols
            num_inputs = len(free_symbols) if free_symbols else 1

        for i, mr in enumerate(mrs):
            is_proven, error_msg = self.prove_mr_with_expr(
                mr=mr,
                sympy_expr=sympy_expr,
                num_inputs=num_inputs,
                operator_name=operator_name,
            )

            if is_proven:
                proven_mrs.append(mr)
                self.logger.debug(f"MR {i+1} proven: {mr.description}")
            else:
                self.logger.debug(
                    f"MR {i+1} proof failed: {mr.description}. Reason: {error_msg}"
                )

        log_structured(
            self.logger,
            "GEN",
            f"SymPy proof completed: {len(proven_mrs)}/{len(mrs)} MRs proven",
        )
        return proven_mrs
