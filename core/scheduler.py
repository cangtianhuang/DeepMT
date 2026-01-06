"""
任务调度器：协调MR生成、插件执行和结果管理
支持使用预生成的MR（通过MR知识库）或动态生成MR
"""

from typing import Any, List, Optional

from core.framework import FrameworkType
from core.ir_manager import IRManager
from core.logger import get_logger
from core.plugins_manager import PluginsManager
from core.results_manager import ResultsManager
from core.test_runner import TestRunner
from ir.schema import ApplicationIR, MetamorphicRelation, ModelIR, OperatorIR


class TaskScheduler:
    """任务调度器：协调整个测试流程"""

    def __init__(
        self,
        ir_manager: IRManager,
        mr_generator: Any,
        plugins_manager: PluginsManager,
        results_manager: ResultsManager,
    ):
        """
        初始化任务调度器

        Args:
            ir_manager: IR管理器
            mr_generator: MR生成器（可以是OperatorMRGenerator等）
            plugins_manager: 插件管理器
            results_manager: 结果管理器
        """
        self.ir_manager = ir_manager
        self.mr_generator = mr_generator
        self.plugins_manager = plugins_manager
        self.results_manager = results_manager
        self.logger = get_logger(self.__class__.__name__)

    def run_task(
        self,
        ir_object: Any,
        target_framework: FrameworkType,
        pre_generated_mrs: Optional[List[MetamorphicRelation]] = None,
    ):
        """
        运行测试任务

        Args:
            ir_object: IR对象（OperatorIR, ModelIR, 或 ApplicationIR）
            target_framework: 目标框架名称（如 "pytorch", "tensorflow", "paddlepaddle"）
            pre_generated_mrs: 预生成的MR列表（可选，如果提供则跳过MR生成）
        """
        self.logger.info(
            f"Starting task for {type(ir_object).__name__}: {ir_object.name if hasattr(ir_object, 'name') else 'unknown'}"
        )
        self.logger.info(f"Target framework: {target_framework}")

        try:
            # 验证IR对象
            if not self.ir_manager.validate_ir(ir_object):
                self.logger.error("IR validation failed")
                return

            # 获取MR列表（使用预生成的或动态生成）
            if pre_generated_mrs is not None:
                self.logger.info(f"Using {len(pre_generated_mrs)} pre-generated MRs")
                mrs = pre_generated_mrs
            else:
                # 动态生成蜕变关系
                self.logger.info("Generating metamorphic relations...")
                if self.mr_generator is None:
                    raise ValueError(
                        "MR generator not provided and no pre-generated MRs"
                    )
                mrs = self.mr_generator.generate(ir_object)

                if not mrs:
                    self.logger.warning("No MRs generated")
                    return

                self.logger.info(f"Generated {len(mrs)} MRs")

            # 获取插件
            try:
                plugin = self.plugins_manager.get_plugin(target_framework)
            except KeyError as e:
                self.logger.error(f"Plugin not found: {e}")
                return

            # 执行每个MR
            results = []
            for i, mr in enumerate(mrs):
                self.logger.info(f"Executing MR {i+1}/{len(mrs)}: {mr.description}")

                try:
                    # 将IR和MR转换为框架代码
                    run_func = plugin.ir_to_code(ir_object, mr)

                    # 执行代码
                    output = plugin.execute(run_func)

                    # 存储结果
                    results.append((mr, output))
                    self.logger.debug(f"MR {i+1} executed successfully")

                except Exception as e:
                    self.logger.error(f"Error executing MR {i+1}: {e}")
                    # 继续执行其他MR
                    continue

            # 比对和存储结果
            if results:
                self.logger.info("Comparing and storing results...")
                self.results_manager.compare_and_store(ir_object, results)
                self.logger.info("Task completed successfully")
            else:
                self.logger.warning("No results to store")

        except Exception as e:
            self.logger.error(f"Task failed with error: {e}")
            raise
