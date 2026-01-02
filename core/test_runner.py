"""
测试执行器：使用预生成的MR执行测试
实现MR生成与测试的分离
"""

from typing import Any, List, Tuple

from core.framework import FrameworkType
from core.logger import get_logger
from core.plugins_manager import PluginsManager
from core.results_manager import ResultsManager
from ir.schema import ApplicationIR, MetamorphicRelation, ModelIR, OperatorIR


class TestRunner:
    """
    测试执行器：使用预生成的MR执行测试

    与TaskScheduler的区别：
    - TaskScheduler：生成MR + 执行测试（耦合）
    - TestRunner：只执行测试，使用预生成的MR（分离）
    """

    def __init__(
        self, plugins_manager: PluginsManager, results_manager: ResultsManager
    ):
        """
        初始化测试执行器

        Args:
            plugins_manager: 插件管理器
            results_manager: 结果管理器
        """
        self.plugins_manager = plugins_manager
        self.results_manager = results_manager
        self.logger = get_logger()

    def run_with_mrs(
        self,
        ir_object: Any,
        mrs: List[MetamorphicRelation],
        target_framework: FrameworkType,
    ):
        """
        使用预生成的MR执行测试

        Args:
            ir_object: IR对象（OperatorIR, ModelIR, 或 ApplicationIR）
            mrs: 预生成的MR列表
            target_framework: 目标框架名称（"pytorch", "tensorflow", "paddlepaddle"）
        """
        self.logger.info(
            f"Running tests for {type(ir_object).__name__}: {ir_object.name if hasattr(ir_object, 'name') else 'unknown'}"
        )
        self.logger.info(f"Target framework: {target_framework}")
        self.logger.info(f"Using {len(mrs)} pre-generated MRs")

        try:
            # 验证IR对象
            if not isinstance(ir_object, (OperatorIR, ModelIR, ApplicationIR)):
                raise ValueError(f"Invalid IR type: {type(ir_object)}")

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
                self.logger.info("Test execution completed successfully")
            else:
                self.logger.warning("No results to store")

        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            raise
