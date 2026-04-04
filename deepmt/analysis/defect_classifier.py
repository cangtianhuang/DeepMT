"""
缺陷分类器：比对MR期望与实际输出，检测和分类缺陷
"""

import numpy as np
from typing import Any, Tuple, Dict

from deepmt.ir.schema import MetamorphicRelation
from deepmt.core.logger import get_logger, log_error


class DefectClassifier:
    """缺陷分类器：检测和分类MR违反情况"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def compare(
        self,
        orig_output: Any,
        trans_output: Any,
        mr: MetamorphicRelation,
        tolerance: float = 1e-6,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        比对原始输出和变换后输出，检查MR是否满足

        Args:
            orig_output: 原始输入对应的输出
            trans_output: 变换后输入对应的输出
            mr: 蜕变关系对象
            tolerance: 数值容差

        Returns:
            (is_match, defect_info)
            is_match: True表示MR满足，False表示违反
            defect_info: 缺陷信息字典，包含type和details
        """
        try:
            # 转换为numpy数组进行比较
            orig_arr = self._to_numpy(orig_output)
            trans_arr = self._to_numpy(trans_output)

            # 优先使用 oracle_expr 进行数值验证
            if mr.oracle_expr:
                is_match, defect_info = self._eval_oracle_expr(
                    mr.oracle_expr, orig_arr, trans_arr, tolerance
                )
            else:
                # oracle_expr 为空时默认检查相等
                is_match, defect_info = self._check_equal(orig_arr, trans_arr, tolerance)

            return is_match, defect_info

        except Exception as e:
            log_error(self.logger, "Error in compare", exception=e)
            return False, {
                "type": "EXCEPTION",
                "details": f"Comparison failed: {str(e)}",
            }

    def _to_numpy(self, output: Any) -> np.ndarray:
        """将输出转换为numpy数组"""
        # 如果是numpy数组
        if isinstance(output, np.ndarray):
            return output

        # 如果是PyTorch tensor
        if hasattr(output, "numpy"):
            return output.detach().cpu().numpy()

        # 如果是TensorFlow tensor
        if hasattr(output, "numpy"):
            return output.numpy()

        # 如果是标量
        if isinstance(output, (int, float)):
            return np.array(output)

        # 如果是列表
        if isinstance(output, (list, tuple)):
            return np.array(output)

        # 其他情况
        return np.array(output)

    def _eval_oracle_expr(
        self,
        oracle_expr: str,
        orig: np.ndarray,
        trans: np.ndarray,
        tolerance: float = 1e-6,
    ) -> Tuple[bool, Dict[str, Any]]:
        """使用 oracle_expr 表达式进行数值验证（`==` 自动使用容差比较）

        oracle_expr 中可用变量：
          - orig: 原始输出
          - trans: 变换后输出
          - x: orig 的别名（兼容 SymPy 证明侧的命名）
          - all / any / abs：numpy 对应函数
        """
        # 包装数组：让 == 走 np.isclose，其余算术/比较保持原样
        tol = tolerance

        class _T:
            """浮点容差感知的数组包装器"""
            __slots__ = ("v",)

            def __init__(self, v):
                self.v = np.asarray(v, dtype=float) if not isinstance(v, np.ndarray) else v

            # 算术运算 —— 结果仍为 _T，保持容差传递
            def __add__(self, o):  return _T(self.v + _u(o))
            def __radd__(self, o): return _T(_u(o) + self.v)
            def __sub__(self, o):  return _T(self.v - _u(o))
            def __rsub__(self, o): return _T(_u(o) - self.v)
            def __mul__(self, o):  return _T(self.v * _u(o))
            def __rmul__(self, o): return _T(_u(o) * self.v)
            def __truediv__(self, o):  return _T(self.v / _u(o))
            def __rtruediv__(self, o): return _T(_u(o) / self.v)
            def __neg__(self): return _T(-self.v)
            def __abs__(self): return _T(np.abs(self.v))

            # 比较 —— 返回普通 np.ndarray（bool），可参与 | & 运算
            def __eq__(self, o):  return np.isclose(self.v, _u(o), atol=tol, rtol=0)
            def __lt__(self, o):  return self.v <  _u(o)
            def __le__(self, o):  return self.v <= _u(o)
            def __gt__(self, o):  return self.v >  _u(o)
            def __ge__(self, o):  return self.v >= _u(o)

        def _u(x):
            return x.v if isinstance(x, _T) else x

        # 去掉外层 all(...) 包装，交由末尾的 np.all 统一处理
        expr = oracle_expr.strip()
        if expr.startswith("all(") and expr.endswith(")"):
            expr = expr[4:-1]

        ctx = {
            "__builtins__": {},
            "orig":  _T(orig),
            "trans": _T(trans),
            "x":     _T(orig),
            "all": np.all,
            "any": np.any,
            "abs": lambda v: _T(np.abs(_u(v))),
            "np": np,
        }
        try:
            result = eval(expr, ctx)  # noqa: S307
            raw = _u(result)
            is_match = bool(np.all(raw))
            if is_match:
                return True, {}
            return False, {
                "type": "ORACLE_VIOLATION",
                "details": f"oracle_expr '{oracle_expr}' not satisfied (tolerance={tolerance})",
            }
        except Exception as e:
            return False, {
                "type": "ORACLE_EVAL_ERROR",
                "details": f"Failed to evaluate oracle_expr '{oracle_expr}': {e}",
            }

    def _check_equal(
        self, orig: np.ndarray, trans: np.ndarray, tolerance: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """检查是否相等（数值容差）"""
        try:
            # 检查形状是否一致
            if orig.shape != trans.shape:
                return False, {
                    "type": "SHAPE_MISMATCH",
                    "details": f"Shape mismatch: {orig.shape} vs {trans.shape}",
                }

            # 计算差异
            diff = np.abs(orig - trans)
            max_diff = np.max(diff)

            if max_diff <= tolerance:
                return True, {}
            else:
                return False, {
                    "type": "NUMERICAL_DEVIATION",
                    "details": f"Max difference: {max_diff}, tolerance: {tolerance}",
                }

        except Exception as e:
            return False, {
                "type": "COMPARISON_ERROR",
                "details": f"Error in equal check: {str(e)}",
            }

    def _check_proportional(
        self, orig: np.ndarray, trans: np.ndarray, tolerance: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """检查是否成比例"""
        try:
            if orig.shape != trans.shape:
                return False, {
                    "type": "SHAPE_MISMATCH",
                    "details": f"Shape mismatch: {orig.shape} vs {trans.shape}",
                }

            # 避免除零
            if np.allclose(orig, 0):
                return np.allclose(trans, 0, atol=tolerance), {
                    "type": "PROPORTIONAL_VIOLATION",
                    "details": "Both should be zero for proportional check",
                }

            # 计算比例
            ratio = trans / orig
            if np.allclose(ratio, ratio.flat[0], atol=tolerance):
                return True, {}
            else:
                return False, {
                    "type": "PROPORTIONAL_VIOLATION",
                    "details": f"Not proportional, ratio varies: {np.min(ratio)} to {np.max(ratio)}",
                }

        except Exception as e:
            return False, {
                "type": "COMPARISON_ERROR",
                "details": f"Error in proportional check: {str(e)}",
            }

    def _check_invariant(
        self, orig: np.ndarray, trans: np.ndarray, tolerance: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """检查是否不变（等同于equal）"""
        return self._check_equal(orig, trans, tolerance)

    def _check_monotonic(
        self, orig: np.ndarray, trans: np.ndarray, tolerance: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """检查单调性（简化实现）"""
        # TODO: 实现更复杂的单调性检查
        return self._check_equal(orig, trans, tolerance)

    def _check_custom(
        self,
        orig: np.ndarray,
        trans: np.ndarray,
        mr: MetamorphicRelation,
        tolerance: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        """自定义比对逻辑"""
        # 默认使用相等检查
        return self._check_equal(orig, trans, tolerance)

    def classify(self, mr: MetamorphicRelation, outputs: Tuple[Any, Any]) -> str:
        """
        分类MR测试结果（兼容旧接口）

        Args:
            mr: 蜕变关系
            outputs: (原始输出, 变换后输出)

        Returns:
            "Pass" 或 "Fail"
        """
        orig_output, trans_output = outputs
        is_match, _ = self.compare(orig_output, trans_output, mr)
        return "Pass" if is_match else "Fail"
