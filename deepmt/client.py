"""
DeepMT 主 API 类

面向 Python 程序化调用的高层接口，行为与 CLI 主链对齐：
  - MR 生成：OperatorMRGenerator（四阶段流水线）
  - 测试执行：BatchTestRunner + RandomGenerator（从 input_specs 自动生成输入）
  - 结果查询：ResultsManager

注意：模型层（ModelMR）与应用层（AppMR）尚未实现，不在此 API 中暴露。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from deepmt.core.logger import logger
from deepmt.core.plugins_manager import FrameworkType
from deepmt.core.results_manager import ResultsManager
from deepmt.mr_generator.base.mr_repository import MRRepository


class TestResult:
    """测试结果封装类"""

    def __init__(
        self,
        name: str,
        framework: FrameworkType,
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
    DeepMT 主 API 类

    面向程序化调用的接口，封装批量测试和结果查询。
    推荐通过 CLI 使用全部功能（`deepmt test batch`、`deepmt mr generate` 等）；
    此类提供的是对核心主链的 Python 包装，适合脚本集成和自动化场景。

    示例:
        >>> from deepmt import DeepMT
        >>> client = DeepMT()
        >>> results = client.get_test_history()
        >>> failed = client.get_failed_tests()
    """

    def __init__(
        self,
        db_path: str = "data/results/defects.db",
        mr_repo_dir: str = "data/knowledge/mr_repository/operator",
    ):
        """
        初始化 DeepMT 客户端。

        Args:
            db_path:     测试结果数据库路径（相对于当前工作目录）
            mr_repo_dir: MR 知识库目录路径
        """
        logger.info("Initializing DeepMT client")
        self.results_manager = ResultsManager(db_path=db_path)
        self.mr_repository = MRRepository(repo_dir=mr_repo_dir)

    def run_batch_test(
        self,
        operator_name: str,
        framework: FrameworkType = "pytorch",
        n_samples: int = 10,
        verified_only: bool = True,
    ) -> TestResult:
        """
        对单个算子运行批量蜕变测试（与 `deepmt test batch` CLI 行为对齐）。

        前提：MR 知识库中已有该算子的 MR（通过 `deepmt mr generate` 生成）；
        算子已有 input_specs（通过 `deepmt catalog enrich` 丰富）。

        Args:
            operator_name: 算子全限定名，如 "torch.nn.functional.relu"
            framework:     目标框架
            n_samples:     每条 MR 的随机输入样本数
            verified_only: 是否只使用已验证的 MR

        Returns:
            TestResult 对象
        """
        import time
        from deepmt.ir.schema import OperatorIR
        from deepmt.mr_generator.base.operator_catalog import OperatorCatalog
        from deepmt.analysis.verification.random_generator import RandomGenerator
        from deepmt.engine.batch_test_runner import BatchTestRunner

        start_time = time.time()
        logger.info(f"Running batch test: {operator_name} on {framework}")

        mrs = self.mr_repository.load(operator_name, verified_only=verified_only)
        if not mrs:
            raise ValueError(
                f"No MRs found for '{operator_name}' in repository. "
                f"Run `deepmt mr generate {operator_name} --save` first."
            )

        catalog = OperatorCatalog()
        op_entry = catalog.get(operator_name, framework=framework)
        input_specs = op_entry.get("input_specs", []) if op_entry else []

        operator_ir = OperatorIR(name=operator_name, input_specs=input_specs)
        generator = RandomGenerator(n_samples=n_samples)
        runner = BatchTestRunner(
            results_manager=self.results_manager,
            input_generator=generator,
        )

        runner.run(operator_ir, mrs, framework=framework)

        duration = time.time() - start_time
        summary_rows = self.results_manager.get_summary(operator_name)
        if summary_rows:
            row = summary_rows[0]
            return TestResult(
                name=row["ir_name"],
                framework=framework,
                total_tests=row["total_tests"],
                passed_tests=row["passed_tests"],
                failed_tests=row["failed_tests"],
                duration=duration,
            )
        return TestResult(
            name=operator_name,
            framework=framework,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            duration=duration,
        )

    def get_test_history(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取测试历史摘要。

        Args:
            name: 算子名称（None 表示全部）

        Returns:
            摘要列表，每项含 ir_name / total_tests / passed_tests / failed_tests
        """
        return self.results_manager.get_summary(name)

    def get_failed_tests(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取失败测试用例列表。

        Args:
            limit: 返回数量上限

        Returns:
            失败用例列表
        """
        return self.results_manager.get_failed_tests(limit)
