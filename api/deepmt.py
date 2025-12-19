"""
DeepMT 主API类
用户友好的接口，隐藏IR和内部实现细节
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from ir.converter import IRConverter
from ir.schema import OperatorIR, ModelIR, ApplicationIR
from mr_generator.operator.operator_mr import OperatorMRGenerator
from mr_generator.operator.knowledge_base import KnowledgeBase
from mr_generator.base.mr_repository import MRRepository
from core.scheduler import TaskScheduler
from core.test_runner import TestRunner
from core.ir_manager import IRManager
from core.plugins_manager import PluginsManager
from core.results_manager import ResultsManager
from core.logger import get_logger


class TestResult:
    """测试结果封装类"""

    def __init__(
        self,
        name: str,
        framework: str,
        total_tests: int,
        passed_tests: int,
        failed_tests: int,
        duration: float,
        details: Optional[List[Dict]] = None,
    ):
        self.name = name
        self.framework = framework
        self.total_tests = total_tests
        self.passed_tests = passed_tests
        self.failed_tests = failed_tests
        self.duration = duration
        self.details = details or []

    def summary(self) -> str:
        """返回测试摘要字符串"""
        lines = [
            "=" * 60,
            "DeepMT 测试结果",
            "=" * 60,
            f"名称: {self.name}",
            f"框架: {self.framework}",
            f"总测试数: {self.total_tests}",
            f"通过: {self.passed_tests}",
            f"失败: {self.failed_tests}",
            f"耗时: {self.duration:.2f}s",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "framework": self.framework,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "duration": self.duration,
            "details": self.details,
        }


class DeepMT:
    """
    DeepMT 主API类

    用户友好的接口，隐藏IR和内部实现细节

    示例:
        >>> from deepmt import DeepMT
        >>> client = DeepMT()
        >>> result = client.test_operator("Add", [1.0, 2.0], "pytorch")
        >>> print(result.summary())
    """

    def __init__(self, db_path: str = "data/defects.db", log_level: str = "INFO"):
        """
        初始化DeepMT客户端

        Args:
            db_path: 数据库路径
            log_level: 日志级别
        """
        self.logger = get_logger()
        self.logger.info("Initializing DeepMT client")

        # 初始化核心组件
        self.ir_manager = IRManager()
        self.plugins_manager = PluginsManager()
        self.plugins_manager.load_plugins()
        self.results_manager = ResultsManager(db_path=db_path)

        # IR转换器
        self.ir_converter = IRConverter()

        # MR生成器（延迟初始化）
        self._mr_generator = None

        # MR知识库（可选，用于MR重用）
        self.mr_repository = MRRepository()

    def test_operator(
        self,
        name: str,
        inputs: List[Any],
        framework: str = "pytorch",
        properties: Optional[Dict[str, Any]] = None,
    ) -> TestResult:
        """
        测试算子

        Args:
            name: 算子名称（如 "Add", "Multiply"）
            inputs: 输入值列表
            framework: 目标框架（"pytorch", "tensorflow", "paddle"）
            properties: 算子属性（可选，会自动推断）

        Returns:
            TestResult对象

        示例:
            >>> client = DeepMT()
            >>> result = client.test_operator("Add", [1.0, 2.0], "pytorch")
            >>> print(result.summary())
        """
        import time

        start_time = time.time()

        self.logger.info(f"Testing operator: {name} on {framework}")

        try:
            # 1. 自动创建IR（用户不需要知道IR的存在）
            operator_ir = self.ir_converter.from_operator_name(
                name=name, inputs=inputs, properties=properties
            )

            # 2. 尝试从知识库加载MR，如果没有则生成
            mrs = None
            if self.mr_repository.exists(operator_ir.name):
                self.logger.info(
                    f"Loading MRs from knowledge base for {operator_ir.name}"
                )
                mrs = self.mr_repository.load(operator_ir.name)
            else:
                self.logger.info(f"Generating MRs for {operator_ir.name}")
                if self._mr_generator is None:
                    kb = KnowledgeBase()
                    self._mr_generator = OperatorMRGenerator(kb)
                mrs = self._mr_generator.generate(operator_ir)
                # 保存到知识库以便后续重用
                self.mr_repository.save(operator_ir.name, mrs)

            # 3. 使用TestRunner执行测试（MR生成与测试分离）
            test_runner = TestRunner(
                plugins_manager=self.plugins_manager,
                results_manager=self.results_manager,
            )
            test_runner.run_with_mrs(operator_ir, mrs, framework)

            # 4. 获取结果
            duration = time.time() - start_time
            summary_data = self.results_manager.get_summary(operator_ir.name)

            if summary_data:
                item = summary_data[0]
                result = TestResult(
                    name=item["ir_name"],
                    framework=framework,
                    total_tests=item["total_tests"],
                    passed_tests=item["passed_tests"],
                    failed_tests=item["failed_tests"],
                    duration=duration,
                )
            else:
                result = TestResult(
                    name=name,
                    framework=framework,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    duration=duration,
                )

            return result

        except Exception as e:
            self.logger.error(f"Error testing operator: {e}")
            raise

    def test_operators(
        self, operators: List[Dict[str, Any]], framework: str = "pytorch"
    ) -> List[TestResult]:
        """
        批量测试多个算子

        Args:
            operators: 算子列表，每个元素为 {"name": str, "inputs": List}
            framework: 目标框架

        Returns:
            TestResult列表

        示例:
            >>> operators = [
            ...     {"name": "Add", "inputs": [1.0, 2.0]},
            ...     {"name": "Multiply", "inputs": [3.0, 4.0]}
            ... ]
            >>> results = client.test_operators(operators, "pytorch")
            >>> for r in results:
            ...     print(r.summary())
        """
        results = []
        for op in operators:
            result = self.test_operator(
                name=op["name"],
                inputs=op["inputs"],
                framework=framework,
                properties=op.get("properties"),
            )
            results.append(result)
        return results

    def test_from_config(self, config_path: Union[str, Path]) -> List[TestResult]:
        """
        从配置文件运行测试

        Args:
            config_path: 配置文件路径（YAML格式）

        Returns:
            TestResult列表
        """
        import yaml

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        results = []
        for test_config in config.get("tests", []):
            test_type = test_config.get("type", "operator")
            framework = test_config.get("framework", "pytorch")

            if test_type == "operator":
                result = self.test_operator(
                    name=test_config["name"],
                    inputs=test_config["inputs"],
                    framework=framework,
                    properties=test_config.get("properties"),
                )
                results.append(result)
            else:
                self.logger.warning(f"Unsupported test type: {test_type}")

        return results

    def get_test_history(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取测试历史

        Args:
            name: 算子名称（可选，None表示获取所有）

        Returns:
            测试历史列表
        """
        return self.results_manager.get_summary(name)

    def get_failed_tests(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取失败的测试用例

        Args:
            limit: 返回数量限制

        Returns:
            失败测试列表
        """
        return self.results_manager.get_failed_tests(limit)
