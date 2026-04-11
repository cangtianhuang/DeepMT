"""
监控模块命令行入口

使用方式:
    python -m deepmt.monitoring check     # 运行健康检查
    python -m deepmt.monitoring progress  # 查看开发进度
    python -m deepmt.monitoring all       # 运行所有检查
"""

import sys

from deepmt.monitoring.health_checker import HealthChecker
from deepmt.monitoring.progress_tracker import ProgressTracker


def main():
    if len(sys.argv) < 2:
        print("DeepMT 项目监控工具")
        print()
        print("使用方式:")
        print("  python -m deepmt.monitoring check     运行健康检查")
        print("  python -m deepmt.monitoring progress  查看开发进度")
        print("  python -m deepmt.monitoring all       运行所有检查")
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
