"""
MR快速筛选器（Pre-check）：用随机数快速测试MR猜想
过滤掉明显不满足的MR，减少后续SymPy证明的计算量
"""

import numpy as np
from typing import List, Callable, Any, Tuple
import traceback

from ir.schema import MetamorphicRelation
from core.logger import get_logger


class MRPreChecker:
    """MR快速筛选器"""

    def __init__(self, num_test_cases: int = 5, tolerance: float = 1e-6):
        """
        初始化快速筛选器

        Args:
            num_test_cases: 快速测试用例数量（默认5组）
            tolerance: 数值容差（用于浮点数比较）
        """
        self.num_test_cases = num_test_cases
        self.tolerance = tolerance
        self.logger = get_logger()

    def generate_test_inputs(
        self, original_inputs: List[Any], num_cases: int = None
    ) -> List[List[Any]]:
        """
        生成随机测试输入

        Args:
            original_inputs: 原始输入（用于推断类型和形状）
            num_cases: 生成用例数量（默认使用self.num_test_cases）

        Returns:
            测试输入列表
        """
        if num_cases is None:
            num_cases = self.num_test_cases

        test_cases = []

        for _ in range(num_cases):
            test_input = []
            for inp in original_inputs:
                # 根据输入类型生成随机值
                if isinstance(inp, (int, float)):
                    # 标量：生成随机数
                    if isinstance(inp, int):
                        test_input.append(np.random.randint(-10, 10))
                    else:
                        test_input.append(np.random.uniform(-10.0, 10.0))
                elif isinstance(inp, (list, tuple, np.ndarray)):
                    # 数组/张量：生成随机数组
                    shape = np.array(inp).shape if hasattr(inp, "__len__") else (3,)
                    test_input.append(np.random.uniform(-10.0, 10.0, size=shape))
                else:
                    # 其他类型：使用原值
                    test_input.append(inp)

            test_cases.append(test_input)

        return test_cases

    def check_mr(
        self,
        operator_func: Callable,
        mr: MetamorphicRelation,
        original_inputs: List[Any],
    ) -> Tuple[bool, str]:
        """
        快速检查MR是否满足

        Args:
            operator_func: 算子函数
            mr: 蜕变关系
            original_inputs: 原始输入（用于推断类型）

        Returns:
            (是否满足, 错误信息)
        """
        test_cases = self.generate_test_inputs(original_inputs)

        passed_count = 0
        failed_count = 0
        error_messages = []

        for i, test_input in enumerate(test_cases):
            try:
                # 执行原始输入
                orig_output = self._execute_operator(operator_func, test_input)

                # 应用MR变换
                transformed_inputs = mr.transform(*test_input)

                # 确保transformed_inputs是元组或列表
                if not isinstance(transformed_inputs, (tuple, list)):
                    transformed_inputs = (transformed_inputs,)

                # 执行变换后的输入
                trans_output = self._execute_operator(operator_func, transformed_inputs)

                # 检查是否满足期望关系
                if self._check_expected_relation(
                    orig_output,
                    trans_output,
                    mr.expected,
                    mr.tolerance or self.tolerance,
                ):
                    passed_count += 1
                else:
                    failed_count += 1
                    error_messages.append(
                        f"Test case {i+1} failed: "
                        f"orig={orig_output}, trans={trans_output}"
                    )

            except Exception as e:
                # 执行错误也算作失败
                failed_count += 1
                error_messages.append(f"Test case {i+1} error: {str(e)}")
                self.logger.debug(f"Pre-check error: {traceback.format_exc()}")

        # 如果所有测试用例都通过，则认为MR可能满足
        if failed_count == 0:
            return True, ""
        else:
            # 如果失败率超过50%，认为MR不满足
            failure_rate = failed_count / (passed_count + failed_count)
            if failure_rate > 0.5:
                error_msg = f"High failure rate: {failure_rate:.2%}. Errors: {error_messages[:3]}"
                return False, error_msg
            else:
                # 部分失败，但可能由于数值精度问题，保留
                return True, f"Partial failures: {failed_count}/{len(test_cases)}"

    def _execute_operator(self, operator_func: Callable, inputs: List[Any]) -> Any:
        """
        执行算子函数

        Args:
            operator_func: 算子函数
            inputs: 输入列表

        Returns:
            算子输出
        """
        if len(inputs) == 0:
            return operator_func()
        elif len(inputs) == 1:
            return operator_func(inputs[0])
        elif len(inputs) == 2:
            return operator_func(inputs[0], inputs[1])
        else:
            return operator_func(*inputs)

    def _check_expected_relation(
        self, orig_output: Any, trans_output: Any, expected: str, tolerance: float
    ) -> bool:
        """
        检查输出是否满足期望关系

        Args:
            orig_output: 原始输出
            trans_output: 变换后输出
            expected: 期望关系类型
            tolerance: 数值容差

        Returns:
            是否满足
        """
        try:
            # 转换为numpy数组以便比较
            orig = np.asarray(orig_output)
            trans = np.asarray(trans_output)

            if expected == "equal":
                # 相等关系：orig == trans
                if orig.shape != trans.shape:
                    return False
                return np.allclose(orig, trans, atol=tolerance, rtol=tolerance)

            elif expected == "proportional":
                # 比例关系：orig == k * trans 或 orig == -trans
                if orig.shape != trans.shape:
                    return False
                # 检查是否成比例（包括负比例）
                if np.allclose(orig, 0) and np.allclose(trans, 0):
                    return True
                # 检查 orig == -trans
                if np.allclose(orig, -trans, atol=tolerance, rtol=tolerance):
                    return True
                # 检查 orig == k * trans (k != 0)
                non_zero_mask = trans != 0
                if np.any(non_zero_mask):
                    ratios = orig[non_zero_mask] / trans[non_zero_mask]
                    if np.allclose(ratios, ratios[0], atol=tolerance, rtol=tolerance):
                        return True
                return False

            elif expected == "invariant":
                # 不变关系：orig == trans
                return self._check_expected_relation(
                    orig_output, trans_output, "equal", tolerance
                )

            else:
                # 其他关系类型，默认检查相等
                self.logger.warning(
                    f"Unknown expected type: {expected}, using equal check"
                )
                return self._check_expected_relation(
                    orig_output, trans_output, "equal", tolerance
                )

        except Exception as e:
            self.logger.debug(f"Error checking expected relation: {e}")
            return False

    def filter_mrs(
        self,
        operator_func: Callable,
        mr_candidates: List[MetamorphicRelation],
        original_inputs: List[Any],
    ) -> List[MetamorphicRelation]:
        """
        过滤MR候选列表，保留可能满足的MR

        Args:
            operator_func: 算子函数
            mr_candidates: MR候选列表
            original_inputs: 原始输入

        Returns:
            过滤后的MR列表
        """
        filtered = []

        self.logger.info(f"Pre-checking {len(mr_candidates)} MR candidates...")

        for i, mr in enumerate(mr_candidates):
            is_valid, error_msg = self.check_mr(operator_func, mr, original_inputs)

            if is_valid:
                filtered.append(mr)
                self.logger.debug(f"MR {i+1} passed pre-check: {mr.description}")
            else:
                self.logger.debug(
                    f"MR {i+1} failed pre-check: {mr.description}. "
                    f"Reason: {error_msg}"
                )

        self.logger.info(
            f"Pre-check completed: {len(filtered)}/{len(mr_candidates)} MRs passed"
        )

        return filtered
