"""
健康检查器：检查项目模块的可用性
"""

import importlib
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class HealthStatus(Enum):
    """健康状态"""

    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class CheckResult:
    """检查结果"""

    name: str
    status: HealthStatus
    message: str
    duration_ms: float = 0.0


@dataclass
class HealthReport:
    """健康报告"""

    timestamp: datetime
    results: List[CheckResult]

    @property
    def healthy_count(self) -> int:
        return sum(1 for r in self.results if r.status == HealthStatus.HEALTHY)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if r.status == HealthStatus.WARNING)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.status == HealthStatus.ERROR)

    @property
    def overall_status(self) -> HealthStatus:
        if self.error_count > 0:
            return HealthStatus.ERROR
        if self.warning_count > 0:
            return HealthStatus.WARNING
        return HealthStatus.HEALTHY


class HealthChecker:
    """项目健康检查器"""

    # 需要检查的核心模块
    CORE_MODULES = [
        ("core.scheduler", "任务调度器"),
        ("core.test_runner", "测试执行器"),
        ("core.ir_manager", "IR管理器"),
        ("core.plugins_manager", "插件管理器"),
        ("core.results_manager", "结果管理器"),
        ("core.framework", "框架管理"),
        ("core.logger", "日志系统"),
        ("core.config_loader", "配置加载器"),
    ]

    IR_MODULES = [
        ("ir.schema", "IR数据结构"),
        ("ir.converter", "IR转换器"),
    ]

    MR_GENERATOR_MODULES = [
        ("mr_generator.operator.operator_mr", "算子MR生成器"),
        ("mr_generator.operator.mr_precheck", "MR快速筛选"),
        ("mr_generator.operator.sympy_prover", "SymPy证明器"),
        ("mr_generator.operator.code_translator", "代码翻译器"),
        ("mr_generator.operator.mr_deriver", "MR推导器"),
        ("mr_generator.base.knowledge_base", "知识库"),
        ("mr_generator.base.mr_templates", "MR模板池"),
        ("mr_generator.base.mr_repository", "MR存储库"),
    ]

    TOOLS_MODULES = [
        ("tools.llm.client", "LLM客户端"),
        ("tools.web_search.search_tool", "网络搜索工具"),
        ("tools.web_search.operator_fetcher", "算子信息获取器"),
    ]

    PLUGIN_MODULES = [
        ("plugins.pytorch_plugin", "PyTorch插件"),
    ]

    API_MODULES = [
        ("api.deepmt", "主API"),
    ]

    def __init__(self):
        self.all_modules = (
            self.CORE_MODULES
            + self.IR_MODULES
            + self.MR_GENERATOR_MODULES
            + self.TOOLS_MODULES
            + self.PLUGIN_MODULES
            + self.API_MODULES
        )

    def check_import(self, module_name: str) -> CheckResult:
        """检查模块是否可以导入"""
        start = time.time()
        try:
            importlib.import_module(module_name)
            duration = (time.time() - start) * 1000
            return CheckResult(
                name=module_name,
                status=HealthStatus.HEALTHY,
                message="导入成功",
                duration_ms=duration,
            )
        except ImportError as e:
            duration = (time.time() - start) * 1000
            return CheckResult(
                name=module_name,
                status=HealthStatus.ERROR,
                message=f"导入失败: {e}",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return CheckResult(
                name=module_name,
                status=HealthStatus.WARNING,
                message=f"导入警告: {e}",
                duration_ms=duration,
            )

    def run_all_checks(self) -> HealthReport:
        """运行所有检查"""
        results = []
        for module_name, _ in self.all_modules:
            result = self.check_import(module_name)
            results.append(result)

        return HealthReport(
            timestamp=datetime.now(),
            results=results,
        )

    def print_report(self, report: Optional[HealthReport] = None):
        """打印健康报告"""
        if report is None:
            report = self.run_all_checks()

        status_icons = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.ERROR: "❌",
        }

        print("=" * 60)
        print("DeepMT 项目健康检查报告")
        print("=" * 60)
        print(f"检查时间: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(
            f"总体状态: {status_icons[report.overall_status]} {report.overall_status.value}"
        )
        print(
            f"通过: {report.healthy_count} | 警告: {report.warning_count} | 错误: {report.error_count}"
        )
        print("-" * 60)

        # 按类别分组显示
        categories = [
            ("核心模块", self.CORE_MODULES),
            ("IR模块", self.IR_MODULES),
            ("MR生成器", self.MR_GENERATOR_MODULES),
            ("工具模块", self.TOOLS_MODULES),
            ("插件模块", self.PLUGIN_MODULES),
            ("API模块", self.API_MODULES),
        ]

        for category_name, modules in categories:
            print(f"\n[{category_name}]")
            for module_name, display_name in modules:
                # 查找对应的检查结果
                result = next(
                    (r for r in report.results if r.name == module_name), None
                )
                if result:
                    icon = status_icons[result.status]
                    print(f"  {icon} {display_name} ({module_name})")
                    if result.status != HealthStatus.HEALTHY:
                        print(f"      └─ {result.message}")

        print("\n" + "=" * 60)
