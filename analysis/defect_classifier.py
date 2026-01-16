"""
缺陷分类器：比对MR期望与实际输出，检测和分类缺陷
"""

import numpy as np
from typing import Any, Tuple, Dict

from ir.schema import MetamorphicRelation
from core.logger import get_logger, log_error


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

            # 根据MR的期望类型进行比对
            if mr.expected == "equal":
                is_match, defect_info = self._check_equal(
                    orig_arr, trans_arr, tolerance
                )
            elif mr.expected == "proportional":
                is_match, defect_info = self._check_proportional(
                    orig_arr, trans_arr, tolerance
                )
            elif mr.expected == "invariant":
                is_match, defect_info = self._check_invariant(
                    orig_arr, trans_arr, tolerance
                )
            elif mr.expected == "monotonic":
                is_match, defect_info = self._check_monotonic(
                    orig_arr, trans_arr, tolerance
                )
            else:
                # 自定义比对逻辑
                is_match, defect_info = self._check_custom(
                    orig_arr, trans_arr, mr, tolerance
                )

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
