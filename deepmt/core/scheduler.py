"""
任务调度器：协调 MR 生成、插件执行和结果管理
支持使用预生成的 MR（通过 MR 知识库）或动态生成 MR
"""

from typing import Any, List, Optional, Tuple

from deepmt.analysis.mr_verifier import MRVerifier
from deepmt.core.framework import FrameworkType
from deepmt.core.ir_manager import IRManager
from deepmt.core.logger import logger
from deepmt.core.plugins_manager import PluginsManager
from deepmt.core.results_manager import ResultsManager
from deepmt.core.test_runner import TestRunner
from deepmt.ir.schema import ApplicationIR, MetamorphicRelation, ModelIR, OracleResult, OperatorIR


class TaskScheduler:
    """任务调度器：协调整个测试流程"""

    def __init__(
        self,
        ir_manager: IRManager,
        mr_generator: Any,
        plugins_manager: PluginsManager,
        results_manager: ResultsManager,
    ):
        self.ir_manager = ir_manager
        self.mr_generator = mr_generator
        self.plugins_manager = plugins_manager
        self.results_manager = results_manager
        self.verifier = MRVerifier()

    def run_task(
        self,
        ir_object: Any,
        target_framework: FrameworkType,
        pre_generated_mrs: Optional[List[MetamorphicRelation]] = None,
    ):
        """
        运行测试任务。

        Args:
            ir_object:          IR 对象（OperatorIR / ModelIR / ApplicationIR）
            target_framework:   目标框架
            pre_generated_mrs:  预生成的 MR 列表（可选，提供则跳过 MR 生成）
        """
        logger.info(
            f"Starting task for {type(ir_object).__name__}: "
            f"{ir_object.name if hasattr(ir_object, 'name') else 'unknown'}"
        )
        logger.info(f"Target framework: {target_framework}")

        try:
            if not self.ir_manager.validate_ir(ir_object):
                logger.error("IR validation failed")
                return

            if pre_generated_mrs is not None:
                logger.info(f"Using {len(pre_generated_mrs)} pre-generated MRs")
                mrs = pre_generated_mrs
            else:
                logger.info("Generating metamorphic relations...")
                if self.mr_generator is None:
                    raise ValueError(
                        "MR generator not provided and no pre-generated MRs"
                    )
                mrs = self.mr_generator.generate(ir_object)
                if not mrs:
                    logger.warning("No MRs generated")
                    return
                logger.info(f"Generated {len(mrs)} MRs")

            try:
                plugin = self.plugins_manager.get_plugin(target_framework)
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
                logger.info("Task completed successfully")
            else:
                logger.warning("No results to store")

        except Exception as e:
            logger.error(f"Task failed with error: {e}")
            raise
