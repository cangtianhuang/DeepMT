"""
MR快速筛选器（Pre-check）：用随机数快速测试MR猜想
过滤掉明显不满足的MR，减少后续SymPy证明的计算量

设计要求：
- 生成随机测试用例（支持标量、NumPy数组、PyTorch Tensor）
- 通过率 > 80% 时保留MR
- 仅对候选MR执行，已验证MR跳过
"""

import traceback
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from core.logger import get_logger
from ir.schema import MetamorphicRelation

# 尝试导入 PyTorch
try:
    import torch
    from torch import Tensor as TorchTensor

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    TorchTensor = None  # type: ignore


class MRPreChecker:
    """
    MR快速筛选器

    使用随机测试用例快速过滤明显不满足的MR，
    减少后续SymPy证明的计算量。

    通过率阈值：80%（即失败率 < 20%）
    """

    # 通过率阈值（设计文档规定 > 80%）
    PASS_RATE_THRESHOLD = 0.8

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
        self, original_inputs: List[Any], num_cases: Optional[int] = None
    ) -> List[List[Any]]:
        """
        生成随机测试输入

        支持的输入类型：
        - 标量（int, float）
        - NumPy数组（np.ndarray）
        - PyTorch张量（torch.Tensor）
        - 列表/元组

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
                generated = self._generate_random_value(inp)
                test_input.append(generated)

            test_cases.append(test_input)

        return test_cases

    def _generate_random_value(self, original: Any) -> Any:
        """
        根据原始输入生成随机值

        Args:
            original: 原始输入值

        Returns:
            同类型的随机值
        """
        # PyTorch Tensor
        if HAS_TORCH and TorchTensor is not None and isinstance(original, TorchTensor):
            import torch as th

            shape = original.shape
            dtype = original.dtype
            device = original.device
            # 生成相同形状、类型、设备的随机Tensor
            if dtype in (th.float32, th.float64, th.float16):
                return th.randn(shape, dtype=dtype, device=device) * 10.0
            elif dtype in (th.int32, th.int64):
                return th.randint(-10, 10, shape, dtype=dtype, device=device)
            else:
                return th.randn(shape, device=device) * 10.0

        # NumPy数组
        if isinstance(original, np.ndarray):
            shape = original.shape
            dtype = original.dtype
            if np.issubdtype(dtype, np.integer):
                return np.random.randint(-10, 10, size=shape).astype(dtype)
            else:
                return np.random.uniform(-10.0, 10.0, size=shape).astype(dtype)

        # 列表/元组：转为数组处理
        if isinstance(original, (list, tuple)):
            arr = np.array(original)
            random_arr = np.random.uniform(-10.0, 10.0, size=arr.shape)
            return type(original)(random_arr.flatten().tolist())

        # 标量
        if isinstance(original, int):
            return np.random.randint(-10, 10)
        if isinstance(original, float):
            return np.random.uniform(-10.0, 10.0)

        # 其他类型：使用原值
        return original

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

                # 获取第一个输入（用于 first_input 检查）
                first_input = test_input[0] if test_input else None

                # 检查是否满足期望关系
                if self._check_expected_relation(
                    orig_output=orig_output,
                    trans_output=trans_output,
                    expected=mr.expected,
                    tolerance=mr.tolerance or self.tolerance,
                    orig_input=first_input,
                    operator_func=operator_func,
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

        # 计算通过率
        total_count = passed_count + failed_count
        if total_count == 0:
            return False, "No test cases executed"

        pass_rate = passed_count / total_count

        # 通过率 > 80% 时保留MR（设计文档规定）
        if pass_rate >= self.PASS_RATE_THRESHOLD:
            if failed_count == 0:
                return True, ""
            else:
                return True, f"Partial failures: {failed_count}/{total_count}"
        else:
            error_msg = (
                f"Low pass rate: {pass_rate:.2%} "
                f"(threshold: {self.PASS_RATE_THRESHOLD:.0%}). "
                f"Errors: {error_messages[:3]}"
            )
            return False, error_msg

    def _execute_operator(self, operator_func: Callable, inputs) -> Any:
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
        self,
        orig_output: Any,
        trans_output: Any,
        expected: str,
        tolerance: float,
        orig_input: Any = None,
        operator_func: Optional[Callable] = None,
    ) -> bool:
        """
        检查输出是否满足期望关系

        支持的关系类型：
        - equal: 原始输出 == 变换后输出
        - proportional: 原始输出 == k * 变换后输出（k为常数）
        - invariant: 同 equal
        - negate: 原始输出 == -变换后输出
        - first_input: 变换后输出 == 第一个输入
        - zero: 变换后输出 == 0
        - idempotent: f(output) == output

        Args:
            orig_output: 原始输出
            trans_output: 变换后输出
            expected: 期望关系类型
            tolerance: 数值容差
            orig_input: 原始输入（用于 first_input 检查）
            operator_func: 算子函数（用于 idempotent 检查）

        Returns:
            是否满足
        """
        try:
            # 转换为numpy数组以便比较
            orig = self._to_numpy(orig_output)
            trans = self._to_numpy(trans_output)

            if expected == "equal" or expected == "invariant":
                # 相等关系：orig == trans
                if orig.shape != trans.shape:
                    return False
                return np.allclose(orig, trans, atol=tolerance, rtol=tolerance)

            elif expected == "negate":
                # 取反关系：orig == -trans
                if orig.shape != trans.shape:
                    return False
                return np.allclose(orig, -trans, atol=tolerance, rtol=tolerance)

            elif expected == "proportional":
                # 比例关系：orig == k * trans（k为非零常数）
                if orig.shape != trans.shape:
                    return False
                # 检查是否成比例（包括负比例）
                if np.allclose(orig, 0) and np.allclose(trans, 0):
                    return True
                # 检查 orig == -trans
                if np.allclose(orig, -trans, atol=tolerance, rtol=tolerance):
                    return True
                # 检查 orig == k * trans (k != 0)
                non_zero_mask = np.abs(trans) > tolerance
                if np.any(non_zero_mask):
                    ratios = orig[non_zero_mask] / trans[non_zero_mask]
                    if len(ratios) > 0 and np.allclose(
                        ratios, ratios[0], atol=tolerance, rtol=tolerance
                    ):
                        return True
                return False

            elif expected == "first_input":
                # 输出等于第一个输入
                if orig_input is None:
                    self.logger.debug("first_input check: orig_input is None")
                    return False
                first_inp = self._to_numpy(orig_input)
                return np.allclose(trans, first_inp, atol=tolerance, rtol=tolerance)

            elif expected == "zero":
                # 输出为零
                return np.allclose(trans, 0, atol=tolerance)

            elif expected == "idempotent":
                # 幂等性：f(f(x)) == f(x)
                # trans_output 已经是 f(x)，需要检查 f(trans_output) == trans_output
                if operator_func is None:
                    self.logger.debug("idempotent check: operator_func is None")
                    return False
                try:
                    nested_output = operator_func(trans_output)
                    nested = self._to_numpy(nested_output)
                    return np.allclose(trans, nested, atol=tolerance, rtol=tolerance)
                except Exception as e:
                    self.logger.debug(f"idempotent check error: {e}")
                    return False

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

    def _to_numpy(self, value: Any) -> np.ndarray:
        """
        将值转换为NumPy数组

        支持 PyTorch Tensor 和其他类型
        """
        if HAS_TORCH and TorchTensor is not None and isinstance(value, TorchTensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

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
