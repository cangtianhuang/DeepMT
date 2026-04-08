"""
MR 验证器：评估蜕变关系的 oracle 表达式，量化测试偏差

职责：
  - 将 oracle_expr（框架无关的数学表达式字符串）在数值上求值
  - 返回包含通过/失败状态、实测差值等完整信息的 OracleResult

设计原则：
  - 框架无关：张量转换委托给 FrameworkPlugin.to_numpy()
  - 容差感知：== 运算符自动走 np.isclose（_T 包装器方案）
  - 可量化：始终记录实测差值，支持精度长期监控
"""

import numpy as np
from typing import Any, Optional

from deepmt.core.logger import logger
from deepmt.ir.schema import MetamorphicRelation, OracleResult


class MRVerifier:
    """MR 验证器：评估 oracle 表达式并量化偏差"""

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
            plugin:   FrameworkPlugin 实例，提供 to_numpy() / allclose()
            x_input:  原始输入张量（oracle_expr 中 x 变量的值）；
                      为 None 时回退使用 orig

        Returns:
            OracleResult
        """
        try:
            orig_np = plugin.to_numpy(orig)
            trans_np = plugin.to_numpy(trans)
            x_np = plugin.to_numpy(x_input) if x_input is not None else orig_np
        except Exception as e:
            return OracleResult(
                passed=False,
                expr=mr.oracle_expr,
                actual_diff=float("inf"),
                tolerance=mr.tolerance,
                detail=f"CONVERSION_ERROR: {e}",
            )

        # 始终测量 orig 与 trans 的最大绝对差（用于监控）
        actual_diff = self._measure_diff(orig_np, trans_np)

        # 形状不一致直接失败
        if orig_np.shape != trans_np.shape:
            return OracleResult(
                passed=False,
                expr=mr.oracle_expr,
                actual_diff=actual_diff,
                tolerance=mr.tolerance,
                detail=f"SHAPE_MISMATCH: {orig_np.shape} vs {trans_np.shape}",
            )

        # oracle_expr 为空：用插件原生 allclose 做等值检查
        if not mr.oracle_expr:
            is_close, measured_diff = plugin.allclose(orig, trans, atol=mr.tolerance)
            return OracleResult(
                passed=is_close,
                expr="orig == trans",
                actual_diff=measured_diff,
                tolerance=mr.tolerance,
                detail="" if is_close else f"NUMERICAL_DEVIATION: max_diff={measured_diff:.6g}",
            )

        return self._eval_oracle_expr(
            orig_np, trans_np, x_np, mr.oracle_expr, mr.tolerance, actual_diff
        )

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _measure_diff(self, orig_np: np.ndarray, trans_np: np.ndarray) -> float:
        """计算两数组间的最大绝对差值"""
        try:
            if orig_np.shape != trans_np.shape:
                return float("inf")
            return float(np.max(np.abs(orig_np.astype(float) - trans_np.astype(float))))
        except Exception:
            return float("inf")

    def _eval_oracle_expr(
        self,
        orig_np: np.ndarray,
        trans_np: np.ndarray,
        x_np: np.ndarray,
        oracle_expr: str,
        tolerance: float,
        actual_diff: float,
    ) -> OracleResult:
        """
        用容差感知的 _T 包装器对 oracle_expr 求值。

        _T 覆写 __eq__ 使 == 走 np.isclose，其余算术/比较原样保留，
        从而使任意 Python 表达式（含 +、*、abs、all 等）均可正确求值。
        """
        tol = tolerance

        class _T:
            """容差感知的数组包装器"""
            __slots__ = ("v",)

            def __init__(self, v):
                self.v = (
                    v.astype(float)
                    if isinstance(v, np.ndarray)
                    else np.asarray(v, dtype=float)
                )

            def __add__(self, o):      return _T(self.v + _u(o))
            def __radd__(self, o):     return _T(_u(o) + self.v)
            def __sub__(self, o):      return _T(self.v - _u(o))
            def __rsub__(self, o):     return _T(_u(o) - self.v)
            def __mul__(self, o):      return _T(self.v * _u(o))
            def __rmul__(self, o):     return _T(_u(o) * self.v)
            def __truediv__(self, o):  return _T(self.v / _u(o))
            def __rtruediv__(self, o): return _T(_u(o) / self.v)
            def __neg__(self):         return _T(-self.v)
            def __abs__(self):         return _T(np.abs(self.v))

            # 比较运算：== 使用 isclose，其余保持数学语义
            def __eq__(self, o):  return np.isclose(self.v, _u(o), atol=tol, rtol=0)
            def __lt__(self, o):  return self.v <  _u(o)
            def __le__(self, o):  return self.v <= _u(o)
            def __gt__(self, o):  return self.v >  _u(o)
            def __ge__(self, o):  return self.v >= _u(o)

        def _u(x):
            return x.v if isinstance(x, _T) else x

        # 剥去外层 all(...)（末尾统一用 np.all 处理）
        expr = oracle_expr.strip()
        if expr.startswith("all(") and expr.endswith(")"):
            expr = expr[4:-1]

        ctx = {
            "__builtins__": {},
            "orig":  _T(orig_np),
            "trans": _T(trans_np),
            "x":     _T(x_np),
            "all":   np.all,
            "any":   np.any,
            "abs":   lambda v: _T(np.abs(_u(v))),
            "np":    np,
        }

        try:
            raw = eval(expr, ctx)  # noqa: S307
            passed = bool(np.all(_u(raw)))
            return OracleResult(
                passed=passed,
                expr=oracle_expr,
                actual_diff=actual_diff,
                tolerance=tolerance,
                detail="" if passed else f"ORACLE_VIOLATION: '{oracle_expr}' not satisfied",
            )
        except Exception as e:
            logger.debug(f"oracle_expr eval error | expr='{oracle_expr}' | {e}")
            return OracleResult(
                passed=False,
                expr=oracle_expr,
                actual_diff=actual_diff,
                tolerance=tolerance,
                detail=f"ORACLE_EVAL_ERROR: {e}",
            )
