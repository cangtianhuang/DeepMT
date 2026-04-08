"""
进度追踪器：追踪项目开发进度
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class ModuleStatus(Enum):
    """模块状态"""

    COMPLETED = "completed"  # 已完成
    IN_PROGRESS = "in_progress"  # 开发中
    PLANNED = "planned"  # 计划中


@dataclass
class ModuleProgress:
    """模块进度"""

    name: str
    path: str
    status: ModuleStatus
    completion: int  # 0-100
    description: str = ""


class ProgressTracker:
    """项目进度追踪器"""

    # 项目模块进度定义
    MODULES: List[ModuleProgress] = [
        # IR层
        ModuleProgress(
            "OperatorIR", "ir/schema.py", ModuleStatus.COMPLETED, 100, "算子中间表示"
        ),
        ModuleProgress(
            "ModelIR", "ir/schema.py", ModuleStatus.COMPLETED, 100, "模型中间表示"
        ),
        ModuleProgress(
            "ApplicationIR", "ir/schema.py", ModuleStatus.COMPLETED, 100, "应用中间表示"
        ),
        ModuleProgress(
            "IR转换器", "ir/converter.py", ModuleStatus.COMPLETED, 100, "IR转换工具"
        ),
        # 核心层
        ModuleProgress(
            "任务调度器",
            "core/scheduler.py",
            ModuleStatus.COMPLETED,
            100,
            "测试任务协调",
        ),
        ModuleProgress(
            "测试执行器",
            "core/test_runner.py",
            ModuleStatus.COMPLETED,
            100,
            "MR测试执行",
        ),
        ModuleProgress(
            "插件管理器",
            "core/plugins_manager.py",
            ModuleStatus.COMPLETED,
            100,
            "框架插件管理",
        ),
        ModuleProgress(
            "结果管理器",
            "core/results_manager.py",
            ModuleStatus.COMPLETED,
            100,
            "测试结果管理",
        ),
        ModuleProgress(
            "框架管理", "core/framework.py", ModuleStatus.COMPLETED, 100, "框架类型定义"
        ),
        # 算子层MR生成
        ModuleProgress(
            "算子MR生成器",
            "mr_generator/operator/operator_mr.py",
            ModuleStatus.COMPLETED,
            100,
            "LLM猜想+SymPy验证",
        ),
        ModuleProgress(
            "LLM MR生成",
            "mr_generator/operator/operator_llm_mr_generator.py",
            ModuleStatus.COMPLETED,
            100,
            "LLM猜想生成",
        ),
        ModuleProgress(
            "MR快速筛选",
            "mr_generator/operator/mr_precheck.py",
            ModuleStatus.COMPLETED,
            100,
            "随机输入验证",
        ),
        ModuleProgress(
            "SymPy验证器",
            "mr_generator/operator/sympy_prover.py",
            ModuleStatus.COMPLETED,
            100,
            "形式化验证",
        ),
        ModuleProgress(
            "代码翻译器",
            "mr_generator/operator/code_translator.py",
            ModuleStatus.COMPLETED,
            100,
            "代码到SymPy",
        ),
        ModuleProgress(
            "MR模板池",
            "mr_generator/base/mr_templates.py",
            ModuleStatus.COMPLETED,
            100,
            "数学变换模板",
        ),
        ModuleProgress(
            "知识库",
            "mr_generator/base/knowledge_base.py",
            ModuleStatus.COMPLETED,
            100,
            "三层知识管理",
        ),
        # 模型层MR生成
        ModuleProgress(
            "模型MR生成器",
            "mr_generator/model/model_mr.py",
            ModuleStatus.IN_PROGRESS,
            30,
            "模型层MR生成",
        ),
        # 应用层MR生成
        ModuleProgress(
            "应用MR生成器",
            "mr_generator/application/app_mr.py",
            ModuleStatus.PLANNED,
            0,
            "应用层MR生成",
        ),
        # 工具层
        ModuleProgress(
            "LLM客户端",
            "tools/llm/client.py",
            ModuleStatus.COMPLETED,
            100,
            "多提供商LLM接口",
        ),
        ModuleProgress(
            "网络搜索工具",
            "tools/web_search/search_tool.py",
            ModuleStatus.COMPLETED,
            100,
            "多源搜索",
        ),
        ModuleProgress(
            "算子信息获取",
            "tools/web_search/operator_fetcher.py",
            ModuleStatus.COMPLETED,
            100,
            "算子文档获取",
        ),
        # 插件
        ModuleProgress(
            "PyTorch插件",
            "plugins/pytorch_plugin.py",
            ModuleStatus.COMPLETED,
            100,
            "PyTorch适配",
        ),
        ModuleProgress(
            "TensorFlow插件",
            "plugins/tensorflow_plugin.py",
            ModuleStatus.PLANNED,
            0,
            "TensorFlow适配",
        ),
        ModuleProgress(
            "PaddlePaddle插件",
            "plugins/paddle_plugin.py",
            ModuleStatus.PLANNED,
            0,
            "PaddlePaddle适配",
        ),
        # API
        ModuleProgress(
            "主API", "api/deepmt.py", ModuleStatus.COMPLETED, 100, "用户友好接口"
        ),
        # 分析
        ModuleProgress(
            "MR验证器",
            "analysis/mr_verifier.py",
            ModuleStatus.COMPLETED,
            100,
            "oracle评估与缺陷量化",
        ),
        ModuleProgress(
            "报告生成器",
            "analysis/report_generator.py",
            ModuleStatus.PLANNED,
            0,
            "测试报告生成",
        ),
        ModuleProgress(
            "可视化", "analysis/visualizer.py", ModuleStatus.PLANNED, 0, "结果可视化"
        ),
    ]

    def get_all_progress(self) -> List[ModuleProgress]:
        """获取所有模块进度"""
        return self.MODULES

    def get_by_status(self, status: ModuleStatus) -> List[ModuleProgress]:
        """按状态筛选模块"""
        return [m for m in self.MODULES if m.status == status]

    def get_overall_completion(self) -> float:
        """计算整体完成度"""
        if not self.MODULES:
            return 0.0
        total = sum(m.completion for m in self.MODULES)
        return total / len(self.MODULES)

    def print_report(self):
        """打印进度报告"""
        status_icons = {
            ModuleStatus.COMPLETED: "✅",
            ModuleStatus.IN_PROGRESS: "🚧",
            ModuleStatus.PLANNED: "📋",
        }

        print("=" * 60)
        print("DeepMT 项目开发进度")
        print("=" * 60)

        completed = self.get_by_status(ModuleStatus.COMPLETED)
        in_progress = self.get_by_status(ModuleStatus.IN_PROGRESS)
        planned = self.get_by_status(ModuleStatus.PLANNED)

        print(f"整体进度: {self.get_overall_completion():.1f}%")
        print(
            f"已完成: {len(completed)} | 进行中: {len(in_progress)} | 计划中: {len(planned)}"
        )
        print("-" * 60)

        # 进度条
        overall = self.get_overall_completion()
        bar_len = 40
        filled = int(bar_len * overall / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"[{bar}] {overall:.1f}%")
        print("-" * 60)

        # 分类显示
        categories = {
            "IR层": [m for m in self.MODULES if m.path.startswith("ir/")],
            "核心层": [m for m in self.MODULES if m.path.startswith("core/")],
            "MR生成器": [m for m in self.MODULES if m.path.startswith("mr_generator/")],
            "工具层": [m for m in self.MODULES if m.path.startswith("tools/")],
            "插件": [m for m in self.MODULES if m.path.startswith("plugins/")],
            "API": [m for m in self.MODULES if m.path.startswith("api/")],
            "分析": [m for m in self.MODULES if m.path.startswith("analysis/")],
        }

        for category, modules in categories.items():
            if modules:
                category_completion = sum(m.completion for m in modules) / len(modules)
                print(f"\n[{category}] ({category_completion:.0f}%)")
                for m in modules:
                    icon = status_icons[m.status]
                    bar_mini = "█" * (m.completion // 10) + "░" * (
                        10 - m.completion // 10
                    )
                    print(f"  {icon} {m.name}: [{bar_mini}] {m.completion}%")

        print("\n" + "=" * 60)

        # 显示进行中和计划中的详情
        if in_progress:
            print("\n🚧 当前进行中:")
            for m in in_progress:
                print(f"   - {m.name}: {m.description}")

        if planned:
            print("\n📋 待开发:")
            for m in planned:
                print(f"   - {m.name}: {m.description}")
