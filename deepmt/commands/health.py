"""
deepmt health — 系统健康检查子命令组

命令:
    check       运行系统健康检查（配置、依赖、数据库等）
    progress    查看项目开发进度
    all         同时运行 check 与 progress
"""

import sys

import click


@click.group()
def health():
    """系统健康检查与开发进度查看。"""


@health.command("check")
def health_check():
    """运行系统健康检查（配置、依赖、数据库连接等）。

    \b
    示例:
      deepmt health check
    """
    try:
        from deepmt.core.health_checker import HealthChecker
        checker = HealthChecker()
        checker.print_report()
    except Exception as e:
        click.echo(click.style(f"健康检查失败: {e}", fg="red"), err=True)
        sys.exit(1)


@health.command("progress")
def health_progress():
    """查看项目各模块开发进度。

    \b
    示例:
      deepmt health progress
    """
    try:
        from deepmt.core.progress_tracker import ProgressTracker
        tracker = ProgressTracker()
        tracker.print_report()
    except Exception as e:
        click.echo(click.style(f"进度查看失败: {e}", fg="red"), err=True)
        sys.exit(1)


@health.command("all")
def health_all():
    """同时运行健康检查与进度报告。

    \b
    示例:
      deepmt health all
    """
    click.echo("\n" + "=" * 60)
    click.echo("DeepMT 系统报告")
    click.echo("=" * 60 + "\n")

    try:
        from deepmt.core.progress_tracker import ProgressTracker
        tracker = ProgressTracker()
        tracker.print_report()
    except Exception as e:
        click.echo(click.style(f"进度报告失败: {e}", fg="red"), err=True)

    click.echo()

    try:
        from deepmt.core.health_checker import HealthChecker
        checker = HealthChecker()
        checker.print_report()
    except Exception as e:
        click.echo(click.style(f"健康检查失败: {e}", fg="red"), err=True)
