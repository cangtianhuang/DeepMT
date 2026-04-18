"""
健康检查器：检查 DeepMT 核心模块可导入性、数据目录完整性与配置状态。

使用：
    deepmt health check
"""

import importlib
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class CheckResult:
    name: str
    status: HealthStatus
    message: str
    duration_ms: float = 0.0


@dataclass
class HealthReport:
    timestamp: datetime
    results: List[CheckResult] = field(default_factory=list)

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
    """DeepMT 系统健康检查器。

    检查内容：
    1. 核心模块可导入性（使用完整 deepmt.* 路径）
    2. 数据目录结构完整性
    3. MR 知识库可访问性
    4. 结果数据库可访问性
    5. 配置文件加载状态
    6. 可选依赖（PyTorch、FastAPI、SymPy 等）
    """

    # ── 核心模块（必须可导入）─────────────────────────────────────────────────
    CORE_MODULES: List[Tuple[str, str]] = [
        ("deepmt.core.config_manager",  "配置管理器"),
        ("deepmt.core.logger",           "日志系统"),
        ("deepmt.core.plugins_manager",  "插件管理器"),
        ("deepmt.core.results_manager",  "结果管理器"),
        ("deepmt.ir",                     "IR 数据结构"),
    ]

    ENGINE_MODULES: List[Tuple[str, str]] = [
        ("deepmt.engine.batch_test_runner", "批量测试执行器"),
        ("deepmt.engine.model_test_runner", "模型测试执行器"),
    ]

    MR_MODULES: List[Tuple[str, str]] = [
        ("deepmt.mr_generator.operator.operator_mr_generator",   "算子 MR 生成器"),
        ("deepmt.mr_generator.base.mr_repository",               "MR 知识库"),
        ("deepmt.mr_generator.base.mr_templates",                "MR 模板池"),
        ("deepmt.mr_generator.base.operator_catalog",            "算子目录"),
        ("deepmt.analysis.verification.mr_prechecker",           "MR 快速预检"),
        ("deepmt.analysis.verification.mr_verifier",             "Oracle 验证器"),
        ("deepmt.analysis.verification.random_generator",        "随机输入生成器"),
    ]

    ANALYSIS_MODULES: List[Tuple[str, str]] = [
        ("deepmt.analysis.reporting.report_generator",           "报告生成器"),
        ("deepmt.analysis.reporting.mutation_tester",            "变异测试器"),
        ("deepmt.analysis.reporting.evidence_collector",         "证据包采集器"),
        ("deepmt.analysis.qa.defect_deduplicator",               "缺陷去重器"),
        ("deepmt.analysis.qa.cross_framework_tester",            "跨框架一致性测试器"),
        ("deepmt.experiments.organizer",                         "实验数据组织器"),
    ]

    PLUGIN_MODULES: List[Tuple[str, str]] = [
        ("deepmt.plugins.pytorch_plugin",            "PyTorch 插件"),
        ("deepmt.plugins.numpy_plugin",              "NumPy 参考后端"),
        ("deepmt.plugins.paddle_plugin",             "PaddlePaddle 插件"),
        ("deepmt.plugins.tensorflow_plugin",         "TensorFlow 插件（可选）"),
        ("deepmt.plugins.faulty_pytorch_plugin",     "缺陷注入插件 (PyTorch)"),
        ("deepmt.plugins.faulty_tensorflow_plugin",  "缺陷注入插件 (TensorFlow 骨架)"),
    ]

    # ── 可选模块（缺失时为 WARNING，不影响主链）─────────────────────────────
    OPTIONAL_MODULES: List[Tuple[str, str]] = [
        ("deepmt.tools.llm.client",                     "LLM 客户端"),
        ("deepmt.tools.web_search.operator_fetcher",    "算子文档获取器"),
        ("deepmt.mr_generator.operator.sympy_prover",   "SymPy 证明引擎"),
        ("deepmt.ui.app",                                "Web 仪表盘"),
    ]

    # ── 数据目录（必须存在或可创建）────────────────────────────────────────────
    DATA_DIRS: List[Tuple[str, str]] = [
        ("data/logs",                           "日志目录"),
        ("data/knowledge/mr_repository",        "MR 知识库目录"),
        ("data/knowledge/operator_catalog",     "算子目录"),
        ("data/results",                        "测试结果目录"),
        ("data/results/evidence",               "证据包目录"),
    ]

    def __init__(self) -> None:
        self.all_required = (
            self.CORE_MODULES
            + self.ENGINE_MODULES
            + self.MR_MODULES
            + self.ANALYSIS_MODULES
            + self.PLUGIN_MODULES
        )

    # ── 基础检查方法 ──────────────────────────────────────────────────────────

    def _check_import(self, module_name: str, display_name: str, optional: bool = False) -> CheckResult:
        start = time.time()
        try:
            importlib.import_module(module_name)
            ms = (time.time() - start) * 1000
            return CheckResult(module_name, HealthStatus.HEALTHY, f"{display_name} 导入成功", ms)
        except ImportError as e:
            ms = (time.time() - start) * 1000
            status = HealthStatus.WARNING if optional else HealthStatus.ERROR
            return CheckResult(module_name, status, f"导入失败: {e}", ms)
        except Exception as e:
            ms = (time.time() - start) * 1000
            return CheckResult(module_name, HealthStatus.WARNING, f"导入警告: {e}", ms)

    def _check_dir(self, rel_path: str, display_name: str) -> CheckResult:
        path = Path(rel_path)
        if path.exists():
            return CheckResult(rel_path, HealthStatus.HEALTHY, f"{display_name} 存在")
        try:
            path.mkdir(parents=True, exist_ok=True)
            return CheckResult(rel_path, HealthStatus.HEALTHY, f"{display_name} 已创建")
        except OSError as e:
            return CheckResult(rel_path, HealthStatus.ERROR, f"无法创建目录: {e}")

    def _check_config(self) -> CheckResult:
        try:
            from deepmt.core.config_manager import get_config_path
            p = get_config_path()
            if p and p.exists():
                return CheckResult("config", HealthStatus.HEALTHY, f"配置文件已加载: {p.name}")
            return CheckResult("config", HealthStatus.WARNING, "未找到 config.yaml（将使用默认值）")
        except Exception as e:
            return CheckResult("config", HealthStatus.ERROR, f"配置加载异常: {e}")

    def _check_mr_repository(self) -> CheckResult:
        start = time.time()
        try:
            from deepmt.mr_generator.base.mr_repository import MRRepository
            repo = MRRepository()
            operators = [p.stem for p in repo.repo_dir.glob("*.yaml")]
            ms = (time.time() - start) * 1000
            if operators:
                return CheckResult("mr_repository", HealthStatus.HEALTHY,
                                   f"MR 知识库可访问，已有 {len(operators)} 个算子: {', '.join(operators[:5])}", ms)
            return CheckResult("mr_repository", HealthStatus.WARNING,
                               "MR 知识库为空（请运行 deepmt mr batch-generate）", ms)
        except Exception as e:
            ms = (time.time() - start) * 1000
            return CheckResult("mr_repository", HealthStatus.ERROR, f"MR 知识库访问失败: {e}", ms)

    def _check_results_db(self) -> CheckResult:
        start = time.time()
        try:
            from deepmt.core.results_manager import ResultsManager
            rm = ResultsManager()
            conn = sqlite3.connect(rm.db_path)
            conn.execute("SELECT 1")
            conn.close()
            ms = (time.time() - start) * 1000
            return CheckResult("results_db", HealthStatus.HEALTHY,
                               f"结果数据库可访问: {rm.db_path}", ms)
        except Exception as e:
            ms = (time.time() - start) * 1000
            return CheckResult("results_db", HealthStatus.ERROR, f"结果数据库访问失败: {e}", ms)

    def _check_pytorch(self) -> CheckResult:
        start = time.time()
        try:
            import torch
            ms = (time.time() - start) * 1000
            return CheckResult("pytorch", HealthStatus.HEALTHY, f"PyTorch {torch.__version__}", ms)
        except ImportError:
            ms = (time.time() - start) * 1000
            return CheckResult("pytorch", HealthStatus.ERROR, "PyTorch 未安装（核心依赖，请 pip install torch）", ms)

    # ── Phase O 深度检查 ───────────────────────────────────────────────────────

    def _framework_version_matrix(self) -> List[Tuple[str, str, str]]:
        """
        返回 [(framework, version, availability)] 元组列表。
        availability ∈ {"installed", "uninstalled"}.
        """
        from deepmt.plugins import PLUGIN_REGISTRY

        out: List[Tuple[str, str, str]] = []
        for entry in PLUGIN_REGISTRY:
            available = entry.is_available()
            try:
                cls = entry.load_class()
                version = cls.framework_version()
            except Exception:
                version = "error"
            out.append((entry.name, version, "installed" if available else "uninstalled"))
        return out

    def _check_plugin_contract(self) -> List[CheckResult]:
        """
        反射校验 PLUGIN_REGISTRY 中每个插件是否实现 FrameworkPlugin 必需原语。
        仅对 is_available() 为 True 的插件实例化；否则视为可选跳过。
        """
        from deepmt.plugins import PLUGIN_REGISTRY
        from deepmt.plugins.framework_plugin import FrameworkPlugin

        required = [
            "_to_tensor", "_execute_operator", "to_numpy", "get_shape",
            "make_tensor", "allclose", "eval_expr", "element_compare",
        ]
        results: List[CheckResult] = []
        for entry in PLUGIN_REGISTRY:
            key = f"contract.{entry.name}"
            if not entry.is_available():
                results.append(CheckResult(
                    key, HealthStatus.WARNING,
                    f"{entry.name}: framework uninstalled, contract check skipped",
                ))
                continue
            try:
                cls = entry.load_class()
                if not issubclass(cls, FrameworkPlugin):
                    results.append(CheckResult(
                        key, HealthStatus.ERROR,
                        f"{entry.name}: {cls.__name__} 未继承 FrameworkPlugin",
                    ))
                    continue
                missing = [m for m in required if getattr(cls, m, None) is None]
                if missing:
                    results.append(CheckResult(
                        key, HealthStatus.ERROR,
                        f"{entry.name}: 缺失原语 {missing}",
                    ))
                    continue
                # 尝试实例化（会触发 _require_* 检查）
                cls()
                results.append(CheckResult(
                    key, HealthStatus.HEALTHY,
                    f"{entry.name}: 契约完整（{len(required)} 原语）",
                ))
            except Exception as e:
                results.append(CheckResult(
                    key, HealthStatus.ERROR,
                    f"{entry.name}: 契约校验失败 {e}",
                ))
        return results

    def _check_mr_plugin_reachability(self) -> List[CheckResult]:
        """
        对知识库中每个算子，检查其在已装载的核心参考插件中是否可达。
        仅检查 pytorch / numpy / paddlepaddle，不要求所有插件都可达。
        不可达时 WARN，不 ERROR。
        """
        from deepmt.mr_generator.base.mr_repository import MRRepository
        from deepmt.plugins import PLUGIN_REGISTRY

        results: List[CheckResult] = []
        try:
            repo = MRRepository()
            operators = sorted({p.stem for p in repo.repo_dir.glob("*.yaml")})
        except Exception as e:
            return [CheckResult("reachability", HealthStatus.ERROR, f"知识库访问失败: {e}")]

        available_plugins = []
        for entry in PLUGIN_REGISTRY:
            if entry.name in ("pytorch", "numpy", "paddlepaddle") and entry.is_available():
                try:
                    available_plugins.append((entry.name, entry.load_class()()))
                except Exception:
                    continue

        for op in operators:
            gaps = []
            for name, plugin in available_plugins:
                try:
                    plugin._resolve_operator(op)
                except Exception:
                    gaps.append(name)
            if gaps:
                results.append(CheckResult(
                    f"reach.{op}", HealthStatus.WARNING,
                    f"算子 '{op}' 未在插件映射: {gaps}",
                ))
            else:
                results.append(CheckResult(
                    f"reach.{op}", HealthStatus.HEALTHY,
                    f"算子 '{op}' 全部可达",
                ))
        return results

    def _check_catalog_gap(self) -> CheckResult:
        """知识库有 MR 但算子目录未收录时给 WARN。"""
        try:
            from deepmt.mr_generator.base.mr_repository import MRRepository
            from deepmt.mr_generator.base.operator_catalog import OperatorCatalog

            repo = MRRepository()
            repo_ops = {p.stem for p in repo.repo_dir.glob("*.yaml")}
            catalog = OperatorCatalog()
            catalog_ops = set()
            try:
                catalog_ops = set(catalog.list_operators())  # 尽力尝试
            except Exception:
                pass
            gaps = sorted(repo_ops - catalog_ops)
            if not gaps:
                return CheckResult(
                    "catalog_gap", HealthStatus.HEALTHY,
                    "知识库算子全部在目录中",
                )
            return CheckResult(
                "catalog_gap", HealthStatus.WARNING,
                f"以下算子在知识库中有 MR 但未收录算子目录: {gaps}",
            )
        except Exception as e:
            return CheckResult("catalog_gap", HealthStatus.WARNING, f"目录对账失败: {e}")

    def run_deep_checks(self) -> HealthReport:
        """快速检查 + 插件契约 + MR↔插件可达性 + 目录对账。"""
        base = self.run_all_checks()
        base.results.extend(self._check_plugin_contract())
        base.results.extend(self._check_mr_plugin_reachability())
        base.results.append(self._check_catalog_gap())
        return base

    def compute_reachability_matrix(self) -> Dict[str, Dict[str, bool]]:
        """
        产出 {operator: {framework: bool}} 可达性矩阵。
        算子来源 = MR 知识库 YAML + OperatorCatalog 中各框架的算子短名
        （保证新装环境非空）。framework 范围取自 PLUGIN_REGISTRY 中可用的插件。
        """
        from deepmt.mr_generator.base.mr_repository import MRRepository
        from deepmt.mr_generator.base.operator_catalog import OperatorCatalog
        from deepmt.plugins import PLUGIN_REGISTRY

        repo = MRRepository()
        repo_ops = {p.stem for p in repo.repo_dir.glob("*.yaml")}

        catalog_ops: set = set()
        try:
            catalog = OperatorCatalog()
            for fw in catalog.get_all_frameworks():
                for entry in catalog.get_all_entries(fw):
                    catalog_ops.add(entry.name.rsplit(".", 1)[-1])
        except Exception:
            catalog_ops = set()

        operators = sorted(repo_ops | catalog_ops)
        matrix: Dict[str, Dict[str, bool]] = {}

        plugins: List[Tuple[str, Any]] = []
        for entry in PLUGIN_REGISTRY:
            if not entry.is_available():
                continue
            try:
                plugins.append((entry.name, entry.load_class()()))
            except Exception:
                continue

        for op in operators:
            row: Dict[str, bool] = {}
            for name, plugin in plugins:
                try:
                    plugin._resolve_operator(op)
                    row[name] = True
                except Exception:
                    row[name] = False
            matrix[op] = row
        return matrix

    # ── 主检查入口 ─────────────────────────────────────────────────────────────

    def run_all_checks(self) -> HealthReport:
        results: List[CheckResult] = []

        # 配置文件
        results.append(self._check_config())

        # PyTorch（核心依赖）
        results.append(self._check_pytorch())

        # 核心模块
        for mod, name in self.all_required:
            results.append(self._check_import(mod, name, optional=False))

        # 可选模块
        for mod, name in self.OPTIONAL_MODULES:
            results.append(self._check_import(mod, name, optional=True))

        # 数据目录
        for rel_path, name in self.DATA_DIRS:
            results.append(self._check_dir(rel_path, name))

        # 运行时检查
        results.append(self._check_mr_repository())
        results.append(self._check_results_db())

        return HealthReport(timestamp=datetime.now(), results=results)

    # ── 格式化输出 ──────────────────────────────────────────────────────────────

    def print_report(self, report: Optional[HealthReport] = None) -> None:
        if report is None:
            report = self.run_all_checks()

        icons = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️ ",
            HealthStatus.ERROR:   "❌",
        }

        overall_icon = icons[report.overall_status]
        print("=" * 64)
        print("DeepMT 系统健康检查报告")
        print("=" * 64)
        print(f"检查时间: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总体状态: {overall_icon} {report.overall_status.value.upper()}")
        print(f"通过: {report.healthy_count}  警告: {report.warning_count}  错误: {report.error_count}")
        # 框架版本矩阵
        matrix = self._framework_version_matrix()
        if matrix:
            print("─" * 64)
            print("框架运行时版本:")
            for name, version, avail in matrix:
                mark = "✅" if avail == "installed" else "⛔"
                print(f"  {mark} {name:<15} {version}")
        print("─" * 64)

        # 分类展示
        sections = [
            ("环境与配置",  ["config", "pytorch"]),
            ("核心模块",    [m for m, _ in self.CORE_MODULES]),
            ("执行引擎",    [m for m, _ in self.ENGINE_MODULES]),
            ("MR 生成",     [m for m, _ in self.MR_MODULES]),
            ("分析模块",    [m for m, _ in self.ANALYSIS_MODULES]),
            ("插件",        [m for m, _ in self.PLUGIN_MODULES]),
            ("可选依赖",    [m for m, _ in self.OPTIONAL_MODULES]),
            ("数据目录",    [p for p, _ in self.DATA_DIRS]),
            ("运行时",      ["mr_repository", "results_db"]),
        ]

        # 追加深度检查的结果
        all_known_keys = {k for _, keys in sections for k in keys}
        deep_keys = [r.name for r in report.results if r.name not in all_known_keys]
        contract_keys = [k for k in deep_keys if k.startswith("contract.")]
        reach_keys = [k for k in deep_keys if k.startswith("reach.")]
        misc_keys = [k for k in deep_keys if k not in contract_keys + reach_keys]
        if contract_keys:
            sections.append(("插件契约（深度）", contract_keys))
        if reach_keys:
            sections.append(("算子↔插件可达性（深度）", reach_keys))
        if misc_keys:
            sections.append(("其他深度检查", misc_keys))

        result_map = {r.name: r for r in report.results}

        for section_name, keys in sections:
            section_results = [result_map[k] for k in keys if k in result_map]
            if not section_results:
                continue

            all_ok = all(r.status == HealthStatus.HEALTHY for r in section_results)
            sec_icon = "✅" if all_ok else ("⚠️ " if all(r.status != HealthStatus.ERROR for r in section_results) else "❌")
            print(f"\n[{section_name}] {sec_icon}")

            for r in section_results:
                icon = icons[r.status]
                # 截短模块路径显示
                short_name = r.name.replace("deepmt.", "")
                if r.duration_ms > 0:
                    print(f"  {icon} {short_name:<50}  ({r.duration_ms:.0f}ms)")
                else:
                    print(f"  {icon} {short_name}")
                if r.status != HealthStatus.HEALTHY:
                    print(f"        └─ {r.message}")

        print("\n" + "=" * 64)

        if report.overall_status == HealthStatus.HEALTHY:
            print("所有核心模块运行正常。")
        elif report.overall_status == HealthStatus.WARNING:
            print("系统可运行，但部分可选组件不可用（见 ⚠️  条目）。")
        else:
            errors = [r for r in report.results if r.status == HealthStatus.ERROR]
            print(f"发现 {len(errors)} 个错误，核心功能可能受影响。")
            print("快速修复：")
            for r in errors[:3]:
                print(f"  • {r.message}")

        print("=" * 64)
