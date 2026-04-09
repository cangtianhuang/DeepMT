"""
MR 验证器：将 oracle_expr 解析为 lhs <op> rhs，统一经由 plugin 完成数值比较

架构：
  oracle_expr → _strip_quantifier → _parse_top_compare → (lhs_code, op, rhs_code)
    - 调用 plugin.eval_expr 在框架张量空间内求值 lhs / rhs
    - op == "=="   → plugin.allclose(lhs, rhs, atol)     框架原生精密比较
    - op 不等式    → plugin.element_compare(lhs, rhs, op) 框架原生逐元素比较
    - 解析失败     → _complex_eval_fallback               numpy 退化路径
"""

import ast
from typing import Any, Optional, Tuple

import numpy as np

from deepmt.core.logger import logger
from deepmt.ir.schema import MetamorphicRelation, OracleResult
from deepmt.plugins.framework_plugin import CompareResult

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


def _compare(lhs: Any, rhs: Any, op: str, plugin: Any, atol: float) -> CompareResult:
    """
    框架原生比较分发：
      op == "==" → plugin.allclose（精密等值，含广播）
      其他不等式  → plugin.element_compare（逐元素，含广播）
    """
    if op == "==":
        return plugin.allclose(lhs, rhs, atol=atol)
    return plugin.element_compare(lhs, rhs, op)


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
    """MR 验证器：将 oracle_expr 解析为 lhs <op> rhs，经由 plugin 完成数值比较"""

    def verify(
        self,
        orig: Any,
        trans: Any,
        mr: MetamorphicRelation,
        plugin: Any,
        x_input: Any = None,
    ) -> OracleResult:
        """
        验证一条 MR 是否对给定输出成立。

        Args:
            orig:     原始算子输出（框架张量）
            trans:    变换后算子输出（框架张量）
            mr:       蜕变关系（含 oracle_expr 和 tolerance）
            plugin:   FrameworkPlugin 实例
            x_input:  原始输入张量（oracle_expr 中 x 变量）；为 None 时回退使用 orig

        Returns:
            OracleResult
        """
        try:
            orig_shape = plugin.get_shape(orig)
            trans_shape = plugin.get_shape(trans)
        except Exception as e:
            return OracleResult(
                passed=False,
                expr=mr.oracle_expr,
                actual_diff=float("inf"),
                tolerance=mr.tolerance,
                detail=f"SHAPE_CHECK_ERROR: {e}",
            )

        if orig_shape != trans_shape:
            return OracleResult(
                passed=False,
                expr=mr.oracle_expr,
                actual_diff=float("inf"),
                tolerance=mr.tolerance,
                detail=f"SHAPE_MISMATCH: {orig_shape} vs {trans_shape}",
            )

        x = x_input if x_input is not None else orig
        display_expr = mr.oracle_expr or "orig == trans"
        inner = _strip_quantifier(display_expr)
        parsed = _parse_top_compare(inner)

        if parsed is not None:
            lhs_code, op, rhs_code = parsed
            try:
                lhs = plugin.eval_expr(lhs_code, orig, trans, x)
                rhs = plugin.eval_expr(rhs_code, orig, trans, x)
            except Exception as e:
                return OracleResult(
                    passed=False,
                    expr=display_expr,
                    actual_diff=float("inf"),
                    tolerance=mr.tolerance,
                    detail=f"EVAL_ERROR: {e}",
                )
            cmp = _compare(lhs, rhs, op, plugin, mr.tolerance)
            return _to_oracle_result(cmp, display_expr, mr.tolerance, op)

        return self._complex_eval_fallback(inner, plugin, orig, trans, x, mr, display_expr)

    def _complex_eval_fallback(
        self,
        inner: str,
        plugin: Any,
        orig: Any,
        trans: Any,
        x: Any,
        mr: MetamorphicRelation,
        display_expr: str,
    ) -> OracleResult:
        """
        布尔组合表达式退化路径（如 (trans==orig+1)|(x<0)）。
        转换至 numpy 后求值；此路径中的 == 为精确等值，不经 allclose。
        """
        try:
            ctx = {
                "orig": plugin.to_numpy(orig),
                "trans": plugin.to_numpy(trans),
                "x": plugin.to_numpy(x),
            }
            raw = _safe_eval_numpy(inner, ctx)
            passed = bool(np.all(raw))
            return OracleResult(
                passed=passed,
                expr=display_expr,
                actual_diff=float("nan"),
                tolerance=mr.tolerance,
                detail="" if passed else f"ORACLE_VIOLATION: '{display_expr}' not satisfied",
            )
        except Exception as e:
            logger.debug(f"complex oracle eval error | expr='{display_expr}' | {e}")
            return OracleResult(
                passed=False,
                expr=display_expr,
                actual_diff=float("inf"),
                tolerance=mr.tolerance,
                detail=f"ORACLE_EVAL_ERROR: {e}",
            )
