"""
MR快速筛选器：使用 TestRunner 进行少量快速测试

设计原则：
- 框架无关：不直接依赖 PyTorch/TensorFlow
- 职责分离：使用统一的 TestRunner 执行测试
- 快速筛选：少量随机用例，快速过滤明显错误的 MR
"""

import uuid
from typing import Any, Callable, List, Dict

from core.framework import FrameworkType
from core.logger import get_logger
from core.oracle_evaluator import OracleEvaluator
from ir.schema import MetamorphicRelation
from core.plugins_manager import PluginsManager
from core.results_manager import ResultsManager


class MRPreChecker:
    """
    MR快速筛选器（重构版）

    使用 TestRunner 进行少量快速测试，过滤明显不满足的 MR。
    完全框架无关，通过插件系统支持多个框架。

    通过率阈值：80%（即失败率 < 20%）
    """

    # 通过率阈值
    PASS_RATE_THRESHOLD = 0.8
    # 快速测试用例数量
    NUM_TEST_CASES = 5

    def __init__(self):
        """初始化快速筛选器"""
        self.logger = get_logger(self.__class__.__name__)
        self.oracle_evaluator = OracleEvaluator()

        self.plugins_manager = PluginsManager()
        self.plugins_manager.load_plugins()
        self.results_manager = ResultsManager()

    def check_mr(
        self,
        operator_func: Callable,
        mr: MetamorphicRelation,
        original_inputs: List[Any],
        framework: FrameworkType = "pytorch",
    ) -> tuple[bool, str]:
        """
        快速检查单个MR是否满足

        使用少量（5组）随机输入进行快速测试。

        Args:
            operator_func: 算子函数
            mr: 蜕变关系
            original_inputs: 原始输入（用于推断类型和形状）

        Returns:
            (是否通过, 详细信息）
        """
        # 生成随机测试用例
        test_cases = self._generate_test_cases(original_inputs, self.NUM_TEST_CASES)

        passed_count = 0
        failed_count = 0
        error_messages = []

        for i, test_input in enumerate(test_cases):
            try:
                # 1. 构建原始 kwargs
                orig_kwargs = self._build_kwargs(test_input)

                # 2. 执行原始输入
                orig_output = operator_func(**orig_kwargs)

                # 3. 应用 MR 变换
                try:
                    transformed_kwargs = mr.transform(orig_kwargs)
                    if not isinstance(transformed_kwargs, dict):
                        failed_count += 1
                        error_messages.append(
                            f"Test case {i+1}: transform returned {type(transformed_kwargs)}, expected dict"
                        )
                        continue
                except Exception as e:
                    failed_count += 1
                    error_messages.append(f"Test case {i+1}: transform error: {str(e)}")
                    self.logger.debug(f"Transform error: {e}")
                    continue

                # 4. 执行变换后的输入
                trans_output = operator_func(**transformed_kwargs)

                # 5. 使用 oracle_expr 验证（框架无关）
                is_satisfied = self._verify_oracle_expr(
                    mr.oracle_expr,
                    orig_output,
                    trans_output,
                    orig_kwargs,
                    transformed_kwargs,
                    framework=framework,
                )

                if is_satisfied:
                    passed_count += 1
                else:
                    failed_count += 1
                    error_messages.append(
                        f"Test case {i+1}: oracle expression '{mr.oracle_expr}' not satisfied"
                    )

            except Exception as e:
                # 执行错误也算作失败
                failed_count += 1
                error_messages.append(f"Test case {i+1}: execution error: {str(e)}")
                self.logger.debug(f"Execution error: {e}")

        # 计算通过率
        total_count = passed_count + failed_count
        if total_count == 0:
            return False, "No test cases executed"

        pass_rate = passed_count / total_count

        # 通过率 >= 80% 时保留MR
        if pass_rate >= self.PASS_RATE_THRESHOLD:
            if failed_count == 0:
                return True, f"All {total_count} tests passed"
            else:
                return True, f"Passed {passed_count}/{total_count} ({pass_rate:.1%})"
        else:
            error_msg = (
                f"Low pass rate: {pass_rate:.1%} "
                f"(threshold: {self.PASS_RATE_THRESHOLD:.0%}). "
                f"Failed: {failed_count}/{total_count}"
            )
            return False, error_msg

    def _verify_oracle_expr(
        self,
        oracle_expr: str,
        orig_output: Any,
        trans_output: Any,
        orig_kwargs: Dict[str, Any],
        trans_kwargs: Dict[str, Any],
        framework: FrameworkType = "pytorch",
    ) -> bool:
        """
        验证 oracle_expr（框架无关）

        使用 OracleEvaluator 将表达式转换为框架特定的代码并执行。

        Args:
            oracle_expr: 框架无关的表达式（如 "orig == trans"）
            orig_output: 原始输出
            trans_output: 变换后输出
            orig_kwargs: 原始输入字典
            trans_kwargs: 变换后输入字典

        Returns:
            验证结果（True/False）
        """
        try:
            # 提取原始输入（假设第一个参数是 'input' 或 'x'）
            x = orig_kwargs.get("input", orig_kwargs.get("x", None))

            # 使用 OracleEvaluator 编译并执行表达式
            return self.oracle_evaluator.evaluate(
                expr=oracle_expr,
                orig=orig_output,
                trans=trans_output,
                x=x,
                framework=framework,
                tolerance=1e-6,  # 使用默认容差
            )

        except Exception as e:
            self.logger.warning(f"Oracle expression evaluation failed: {e}")
            self.logger.debug(f"Expression: {oracle_expr}")
            return False

    def filter_mrs(
        self,
        operator_func: Callable,
        mr_candidates: List[MetamorphicRelation],
        original_inputs: List[Any],
        framework: FrameworkType = "pytorch",
    ) -> List[MetamorphicRelation]:
        """
        过滤MR候选列表，保留可能满足的MR

        Args:
            operator_func: 算子函数
            mr_candidates: MR候选列表
            original_inputs: 原始输入
            framework: 框架名称

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

    def _generate_test_cases(
        self, original_inputs: List[Any], num_cases: int
    ) -> List[List[Any]]:
        """
        生成随机测试用例

        支持的输入类型：
        - 标量（int, float）
        - NumPy 数组（np.ndarray）
        - PyTorch 张量（torch.Tensor）
        - 列表/元组

        Args:
            original_inputs: 原始输入（用于推断类型和形状）
            num_cases: 生成用例数量

        Returns:
            测试输入列表
        """
        import numpy as np

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
        import numpy as np

        # 尝试导入 PyTorch（如果可用）
        try:
            import torch

            HAS_TORCH = True
        except ImportError:
            HAS_TORCH = False

        # PyTorch Tensor
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

    def _build_kwargs(self, inputs: List[Any]) -> Dict[str, Any]:
        """
        将输入列表转换为 kwargs 字典

        假设：
        - 第一个参数名为 "input" 或 "x"
        - 其他参数名为 "arg1", "arg2", ...

        Args:
            inputs: 输入列表

        Returns:
            kwargs 字典
        """
        if len(inputs) == 0:
            return {}
        elif len(inputs) == 1:
            return {"input": inputs[0]}
        else:
            kwargs = {"input": inputs[0]}
            for i, val in enumerate(inputs[1:], 1):
                kwargs[f"arg{i}"] = val
            return kwargs
