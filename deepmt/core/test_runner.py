"""
测试执行器：使用预生成的 MR 执行测试
实现 MR 生成与测试执行的分离
"""

from typing import Any, List, Tuple

from deepmt.analysis.mr_verifier import MRVerifier
from deepmt.core.plugins_manager import FrameworkType, get_plugins_manager
from deepmt.core.logger import logger
from deepmt.core.results_manager import ResultsManager
from deepmt.ir.schema import ApplicationIR, MetamorphicRelation, ModelIR, OracleResult, OperatorIR


class TestRunner:
    """
    测试执行器：使用预生成的 MR 执行测试

    与 TaskScheduler 的区别：
    - TaskScheduler：生成 MR + 执行测试（耦合）
    - TestRunner：只执行测试，使用预生成的 MR（分离）
    """

    def __init__(self, results_manager: ResultsManager):
        self.results_manager = results_manager
        self.verifier = MRVerifier()

    def run_with_mrs(
        self,
        ir_object: Any,
        mrs: List[MetamorphicRelation],
        target_framework: FrameworkType,
    ):
        """
        使用预生成的 MR 执行测试。

        Args:
            ir_object:        IR 对象（OperatorIR / ModelIR / ApplicationIR）
            mrs:              预生成的 MR 列表
            target_framework: 目标框架（"pytorch" / "tensorflow" / "paddlepaddle"）
        """
        logger.info(
            f"Running tests for {type(ir_object).__name__}: "
            f"{ir_object.name if hasattr(ir_object, 'name') else 'unknown'}"
        )
        logger.info(f"Target framework: {target_framework}")
        logger.info(f"Using {len(mrs)} pre-generated MRs")

        try:
            if not isinstance(ir_object, (OperatorIR, ModelIR, ApplicationIR)):
                raise ValueError(f"Invalid IR type: {type(ir_object)}")

            try:
                plugin = get_plugins_manager().get_plugin(target_framework)
            except KeyError as e:
                logger.error(f"Plugin not found: {e}")
                return

            x_input = (
                ir_object.inputs[0]
                if hasattr(ir_object, "inputs") and ir_object.inputs
                else None
            )

            results: List[Tuple[MetamorphicRelation, OracleResult]] = []

            for i, mr in enumerate(mrs):
                logger.info(f"Executing MR {i+1}/{len(mrs)}: {mr.description}")
                try:
                    run_func = plugin.ir_to_code(ir_object, mr)
                    orig_output, trans_output = plugin.execute(run_func)

                    oracle_result = self.verifier.verify(
                        orig_output, trans_output, mr, plugin, x_input=x_input
                    )
                    results.append((mr, oracle_result))
                    logger.debug(
                        f"MR {i+1}: {'PASS' if oracle_result.passed else 'FAIL'} "
                        f"(diff={oracle_result.actual_diff:.4g})"
                    )
                except Exception as e:
                    logger.error(f"Error executing MR {i+1}: {e}")

            if results:
                self.results_manager.store_result(ir_object, results, str(target_framework))
                logger.info("Test execution completed successfully")
            else:
                logger.warning("No results to store")

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            raise
