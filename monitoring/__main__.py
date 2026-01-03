"""
监控模块命令行入口

使用方式:
    python -m monitoring check     # 运行健康检查
    python -m monitoring progress  # 查看开发进度
    python -m monitoring all       # 运行所有检查
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from monitoring.health_checker import HealthChecker
from monitoring.progress_tracker import ProgressTracker


def main():
    if len(sys.argv) < 2:
        print("DeepMT 项目监控工具")
        print()
        print("使用方式:")
        print("  python -m monitoring check     运行健康检查")
        print("  python -m monitoring progress  查看开发进度")
        print("  python -m monitoring all       运行所有检查")
        return

    command = sys.argv[1].lower()

    if command == "check":
        checker = HealthChecker()
        checker.print_report()

    elif command == "progress":
        tracker = ProgressTracker()
        tracker.print_report()

    elif command == "all":
        print("\n" + "=" * 60)
        print("DeepMT 项目健康监控 - 完整报告")
        print("=" * 60 + "\n")

        # 进度报告
        tracker = ProgressTracker()
        tracker.print_report()

        print("\n")

        # 健康检查
        checker = HealthChecker()
        checker.print_report()

    else:
        print(f"未知命令: {command}")
        print("可用命令: check, progress, all")


if __name__ == "__main__":
    main()
