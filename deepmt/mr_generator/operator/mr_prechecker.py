"""
MR 快速筛选器：对候选 MR 进行少量随机数值测试

设计原则：
- 框架无关：通过插件系统支持多个框架
- 职责分离：transform 绑定由 FrameworkAdapter 负责，
            oracle 评估由 MRVerifier 负责
- 快速筛选：5 组随机用例，通过率 ≥ 80% 保留
"""

from typing import Any, Callable, Dict, List

from deepmt.analysis.mr_verifier import MRVerifier
from deepmt.core.framework import FrameworkType
from deepmt.core.logger import logger
from deepmt.core.plugins_manager import PluginsManager
from deepmt.ir.schema import MetamorphicRelation


class MRPreChecker:
    """MR 快速筛选器"""

    PASS_RATE_THRESHOLD = 0.8
    NUM_TEST_CASES = 5

    def __init__(self):
        self.plugins_manager = PluginsManager()
        self.plugins_manager.load_plugins()
        self.verifier = MRVerifier()

    def check_mr(
        self,
        operator_func: Callable,
        mr: MetamorphicRelation,
        original_inputs: List[Any],
        framework: FrameworkType,
    ) -> tuple[bool, str]:
        """
        快速检查单个 MR 是否满足。

        Args:
            operator_func:   算子函数
            mr:              蜕变关系
            original_inputs: 原始输入（用于推断类型和形状）
            framework:       框架类型

        Returns:
            (是否通过, 详细信息)
        """
        framework_adapter = self.plugins_manager.get_framework_adapter(framework)
        plugin = self.plugins_manager.get_plugin(framework)

        bound_transform = framework_adapter.bind_transform_code(
            mr.transform_code, operator_func
        )
        if bound_transform is None:
            return False, "Failed to bind transform code"

        test_cases = self._generate_test_cases(original_inputs, self.NUM_TEST_CASES)

        passed_count = 0
        failed_count = 0
        error_messages = []

        for i, test_input in enumerate(test_cases):
            try:
                orig_kwargs = self._build_kwargs(test_input)

                orig_output = operator_func(**orig_kwargs)

                try:
                    transformed_kwargs = bound_transform(orig_kwargs)
                    if not isinstance(transformed_kwargs, dict):
                        failed_count += 1
                        error_messages.append(
                            f"Test case {i+1}: transform returned "
                            f"{type(transformed_kwargs)}, expected dict"
                        )
                        continue
                except Exception as e:
                    failed_count += 1
                    error_messages.append(f"Test case {i+1}: transform error: {e}")
                    logger.debug(f"Transform error: {e}")
                    continue

                trans_output = operator_func(**transformed_kwargs)

                # 提取原始输入张量（oracle_expr 中 x 变量）
                x_input = orig_kwargs.get("input", orig_kwargs.get("x", None))

                oracle_result = self.verifier.verify(
                    orig_output, trans_output, mr, plugin, x_input=x_input
                )

                if oracle_result.passed:
                    passed_count += 1
                else:
                    failed_count += 1
                    error_messages.append(
                        f"Test case {i+1}: {oracle_result.detail or 'oracle not satisfied'}"
                    )

            except Exception as e:
                failed_count += 1
                error_messages.append(f"Test case {i+1}: execution error: {e}")
                logger.debug(f"Execution error: {e}")

        total_count = passed_count + failed_count
        if total_count == 0:
            return False, "No test cases executed"

        pass_rate = passed_count / total_count

        if pass_rate >= self.PASS_RATE_THRESHOLD:
            if failed_count == 0:
                return True, f"All {total_count} tests passed"
            return True, f"Passed {passed_count}/{total_count} ({pass_rate:.1%})"
        else:
            return (
                False,
                f"Low pass rate: {pass_rate:.1%} "
                f"(threshold: {self.PASS_RATE_THRESHOLD:.0%}). "
                f"Failed: {failed_count}/{total_count}",
            )

    def filter_mrs(
        self,
        operator_func: Callable,
        mr_candidates: List[MetamorphicRelation],
        original_inputs: List[Any],
        framework: FrameworkType,
    ) -> List[MetamorphicRelation]:
        """
        过滤 MR 候选列表，保留通过快速测试的 MR。

        Args:
            operator_func:  算子函数
            mr_candidates:  MR 候选列表
            original_inputs: 原始输入
            framework:      框架类型

        Returns:
            过滤后的 MR 列表
        """
        if not mr_candidates:
            return []

        filtered = []
        logger.info(f"✅ [CHECK] Pre-checking {len(mr_candidates)} MR candidates...")

        for i, mr in enumerate(mr_candidates):
            is_valid, msg = self.check_mr(operator_func, mr, original_inputs, framework)
            if is_valid:
                filtered.append(mr)
                logger.debug(f"MR {i+1} passed pre-check: {mr.description}")
            else:
                logger.debug(
                    f"MR {i+1} failed pre-check: {mr.description}. Reason: {msg}"
                )

        logger.info(
            f"✅ [CHECK] Pre-check completed: {len(filtered)}/{len(mr_candidates)} MRs passed"
        )
        return filtered

    def _generate_test_cases(
        self, original_inputs: List[Any], num_cases: int
    ) -> List[List[Any]]:
        """生成随机测试用例"""
        test_cases = []
        for _ in range(num_cases):
            test_input = [self._generate_random_value(inp) for inp in original_inputs]
            test_cases.append(test_input)
        return test_cases

    def _generate_random_value(self, original: Any) -> Any:
        """根据原始输入生成同类型随机值"""
        import numpy as np

        try:
            import torch
            HAS_TORCH = True
        except ImportError:
            HAS_TORCH = False

        if HAS_TORCH and "Tensor" in str(type(original)):
            import torch as th
            shape = original.shape
            dtype = original.dtype
            device = original.device
            if dtype in (th.float32, th.float64, th.float16):
                return th.randn(shape, dtype=dtype, device=device) * 10.0
            elif dtype in (th.int32, th.int64):
                return th.randint(-10, 10, shape, dtype=dtype, device=device)
            else:
                return th.randn(shape, device=device) * 10.0

        if isinstance(original, np.ndarray):
            shape = original.shape
            dtype = original.dtype
            if np.issubdtype(dtype, np.integer):
                return np.random.randint(-10, 10, size=shape).astype(dtype)
            return np.random.uniform(-10.0, 10.0, size=shape).astype(dtype)

        if isinstance(original, (list, tuple)):
            arr = np.array(original)
            random_arr = np.random.uniform(-10.0, 10.0, size=arr.shape)
            return type(original)(random_arr.flatten().tolist())

        if isinstance(original, int):
            return int(np.random.randint(-10, 10))
        if isinstance(original, float):
            return float(np.random.uniform(-10.0, 10.0))

        return original

    def _build_kwargs(self, inputs: List[Any]) -> Dict[str, Any]:
        """将输入列表转换为 kwargs 字典"""
        if len(inputs) == 0:
            return {}
        if len(inputs) == 1:
            return {"input": inputs[0]}
        kwargs = {"input": inputs[0]}
        for i, val in enumerate(inputs[1:], 1):
            kwargs[f"arg{i}"] = val
        return kwargs
