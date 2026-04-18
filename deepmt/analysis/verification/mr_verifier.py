"""
MR 验证器：将 oracle_expr 解析为 lhs <op> rhs，统一经由 backend 完成数值比较

架构：
  oracle_expr → _strip_quantifier → _parse_top_compare → (lhs_code, op, rhs_code)
    - 调用 backend.eval_expr 在框架张量空间内求值 lhs / rhs
    - op == "=="   → backend.allclose(lhs, rhs, atol)     框架原生精密比较
    - op 不等式    → backend.element_compare(lhs, rhs, op) 框架原生逐元素比较
    - 解析失败     → _complex_eval_fallback               numpy 退化路径
"""

import ast
import math
from typing import Any, List, Optional, Tuple

import numpy as np

from deepmt.core.logger import logger
from deepmt.ir import MetamorphicRelation, OracleResult
from deepmt.plugins.framework_plugin import CompareResult

# ── 自适应容差计算 ─────────────────────────────────────────────────────────────

# 算子分类到容差策略的映射
_ELEMENTWISE_OPS = frozenset({
    "relu", "sigmoid", "tanh", "leaky_relu", "gelu", "softmax", "log_softmax",
    "elu", "selu", "silu", "abs", "exp", "log", "sqrt", "neg", "sin", "cos",
    "floor", "ceil", "add", "subtract", "divide", "multiply", "pow",
    "batch_norm", "layer_norm", "instance_norm",
})
_REDUCTION_OPS = frozenset({
    "sum", "mean", "max", "min", "std", "var", "prod", "cumsum",
    "cross_entropy", "mse_loss", "binary_cross_entropy",
})
_MATMUL_OPS = frozenset({
    "matmul", "conv2d", "conv1d",
})

_BASE_ATOL = 1e-6
_MATMUL_BASE_ATOL = 1e-5
_REDUCTION_BASE_ATOL = 1e-5
_FP16_MULTIPLIER = 10.0


def calculate_adaptive_tolerance(
    operator_name: str,
    input_shapes: Optional[List[tuple]] = None,
    dtype: str = "float32",
) -> float:
    """
    基于算子类型和输入规模自动计算数值容差阈值。

    规则：
      逐元素算子（relu, sigmoid …）  → atol = 1e-6
      矩阵运算（matmul, conv2d …）   → atol = 1e-5 × √N，N = 最大输入维度乘积
      归约运算（sum, mean …）        → atol = 1e-5 × max_dim
      fp16 dtype                     → 上述结果 × 10
      其他                           → atol = 1e-6

    Args:
        operator_name:  算子名称（与 BenchmarkSuite 一致）
        input_shapes:   输入张量形状列表（可为 None）
        dtype:          输入数据类型字符串

    Returns:
        推荐的绝对容差值 atol
    """
    name = operator_name.lower()
    shapes = input_shapes or []
    fp16 = "16" in dtype

    max_n = 1
    for shape in shapes:
        try:
            n = 1
            for d in shape:
                n *= int(d)
            max_n = max(max_n, n)
        except (TypeError, ValueError):
            pass

    max_dim = max((max(s) for s in shapes if s), default=1)

    if name in _MATMUL_OPS:
        atol = _MATMUL_BASE_ATOL * math.sqrt(max(max_n, 1))
    elif name in _REDUCTION_OPS:
        atol = _REDUCTION_BASE_ATOL * max(max_dim, 1)
    elif name in _ELEMENTWISE_OPS:
        atol = _BASE_ATOL
    else:
        atol = _BASE_ATOL

    if fp16:
        atol *= _FP16_MULTIPLIER

    return float(atol)


# ── AST 操作符映射 ─────────────────────────────────────────────────────────────

_AST_OP_MAP = {
    ast.Eq:    "==",
    ast.NotEq: "!=",
    ast.Lt:    "<",
    ast.LtE:   "<=",
    ast.Gt:    ">",
    ast.GtE:   ">=",
}

# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def _strip_quantifier(expr: str) -> str:
    """用 AST 剥离外层 all(...) / any(...) 包装。"""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return expr
    node = tree.body
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("all", "any")
        and len(node.args) == 1
        and not node.keywords
    ):
        inner = ast.get_source_segment(expr, node.args[0])
        return inner if inner is not None else expr
    return expr


def _parse_top_compare(expr: str) -> Optional[Tuple[str, str, str]]:
    """
    AST 解析顶层单一比较，返回 (lhs_code, op_str, rhs_code)。
    非单一 Compare 节点则返回 None。
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    node = tree.body
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    op_str = _AST_OP_MAP.get(type(node.ops[0]))
    if op_str is None:
        return None
    lhs_code = ast.get_source_segment(expr, node.left)
    rhs_code = ast.get_source_segment(expr, node.comparators[0])
    if lhs_code is None or rhs_code is None:
        return None
    return lhs_code, op_str, rhs_code


def _safe_eval_numpy(code: str, ctx: dict) -> np.ndarray:
    """在受限 numpy 命名空间中对代码求值（仅用于退化路径）。"""
    ns = {
        "__builtins__": {},
        "np": np,
        "abs": np.abs,
        "all": np.all,
        "any": np.any,
        **ctx,
    }
    return np.asarray(eval(code, ns))  # noqa: S307


def _compare(lhs: Any, rhs: Any, op: str, backend: Any, atol: float) -> CompareResult:
    """
    框架原生比较分发：
      op == "==" → backend.allclose（精密等值，含广播）
      其他不等式  → backend.element_compare（逐元素，含广播）
    """
    if op == "==":
        return backend.allclose(lhs, rhs, atol=atol)
    return backend.element_compare(lhs, rhs, op)


def _to_oracle_result(
    cmp: CompareResult,
    expr: str,
    tolerance: float,
    op: str,
) -> OracleResult:
    """将 CompareResult 转换为 OracleResult，附带格式化诊断信息。"""
    if cmp.passed:
        detail = ""
    elif op == "==":
        detail = (
            f"NUMERICAL_DEVIATION: "
            f"max_abs={cmp.max_abs_diff:.6g}, "
            f"max_rel={cmp.max_rel_diff:.6g}, "
            f"mismatched={cmp.mismatched_elements}/{cmp.total_elements} "
            f"({cmp.mismatched_ratio:.1%})"
        )
    else:
        detail = (
            f"INEQUALITY_VIOLATION: {cmp.mismatched_elements}/{cmp.total_elements} "
            f"elements violate '{expr}'; max_abs_diff={cmp.max_abs_diff:.6g}"
        )
    return OracleResult(
        passed=cmp.passed,
        expr=expr,
        actual_diff=cmp.max_abs_diff,
        tolerance=tolerance,
        detail=detail,
        max_rel_diff=cmp.max_rel_diff,
        mismatched_elements=cmp.mismatched_elements,
        total_elements=cmp.total_elements,
    )


# ── 验证器 ────────────────────────────────────────────────────────────────────


class MRVerifier:
    """MR 验证器：将 oracle_expr 解析为 lhs <op> rhs，经由 backend 完成数值比较"""

    def verify(
        self,
        orig: Any,
        trans: Any,
        mr: MetamorphicRelation,
        backend: Any,
        x_input: Any = None,
        operator_name: Optional[str] = None,
        input_shapes: Optional[List[tuple]] = None,
        dtype: str = "float32",
    ) -> OracleResult:
        """
        验证一条 MR 是否对给定输出成立。

        Args:
            orig:           原始算子输出（框架张量）
            trans:          变换后算子输出（框架张量）
            mr:             蜕变关系（含 oracle_expr 和 tolerance）
            backend:        FrameworkPlugin 实例（被测框架的计算后端）
            x_input:        原始输入张量（oracle_expr 中 x 变量）；为 None 时回退使用 orig
            operator_name:  算子名称；提供时自动计算自适应容差（覆盖 mr.tolerance）
            input_shapes:   输入张量形状列表，用于自适应容差计算
            dtype:          输入数据类型字符串，用于 fp16 放宽系数

        Returns:
            OracleResult
        """
        # 优先使用自适应容差；否则退回 MR 模板中的静态容差
        if operator_name is not None:
            atol = calculate_adaptive_tolerance(operator_name, input_shapes, dtype)
        else:
            atol = mr.tolerance

        try:
            orig_shape = backend.get_shape(orig)
            trans_shape = backend.get_shape(trans)
        except Exception as e:
            return OracleResult(
                passed=False,
                expr=mr.oracle_expr,
                actual_diff=float("inf"),
                tolerance=atol,
                detail=f"SHAPE_CHECK_ERROR: {e}",
            )

        if orig_shape != trans_shape:
            return OracleResult(
                passed=False,
                expr=mr.oracle_expr,
                actual_diff=float("inf"),
                tolerance=atol,
                detail=f"SHAPE_MISMATCH: {orig_shape} vs {trans_shape}",
            )

        x = x_input if x_input is not None else orig
        display_expr = mr.oracle_expr or "orig == trans"
        inner = _strip_quantifier(display_expr)
        parsed = _parse_top_compare(inner)

        if parsed is not None:
            lhs_code, op, rhs_code = parsed
            try:
                lhs = backend.eval_expr(lhs_code, orig, trans, x)
                rhs = backend.eval_expr(rhs_code, orig, trans, x)
            except Exception as e:
                return OracleResult(
                    passed=False,
                    expr=display_expr,
                    actual_diff=float("inf"),
                    tolerance=atol,
                    detail=f"EVAL_ERROR: {e}",
                )
            cmp = _compare(lhs, rhs, op, backend, atol)
            return _to_oracle_result(cmp, display_expr, atol, op)

        return self._complex_eval_fallback(inner, backend, orig, trans, x, mr, display_expr, atol)

    def _complex_eval_fallback(
        self,
        inner: str,
        backend: Any,
        orig: Any,
        trans: Any,
        x: Any,
        mr: MetamorphicRelation,
        display_expr: str,
        atol: float = 1e-6,
    ) -> OracleResult:
        """
        布尔组合表达式退化路径（如 (trans==orig+1)|(x<0)）。
        转换至 numpy 后求值；此路径中的 == 为精确等值，不经 allclose。
        """
        try:
            ctx = {
                "orig": backend.to_numpy(orig),
                "trans": backend.to_numpy(trans),
                "x": backend.to_numpy(x),
            }
            raw = _safe_eval_numpy(inner, ctx)
            passed = bool(np.all(raw))
            return OracleResult(
                passed=passed,
                expr=display_expr,
                actual_diff=float("nan"),
                tolerance=atol,
                detail="" if passed else f"ORACLE_VIOLATION: '{display_expr}' not satisfied",
            )
        except Exception as e:
            logger.debug(f"complex oracle eval error | expr='{display_expr}' | {e}")
            return OracleResult(
                passed=False,
                expr=display_expr,
                actual_diff=float("inf"),
                tolerance=atol,
                detail=f"ORACLE_EVAL_ERROR: {e}",
            )
