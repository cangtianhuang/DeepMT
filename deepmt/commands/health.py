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
@click.option("--deep", is_flag=True, help="追加插件契约、算子可达性与目录对账检查")
def health_check(deep: bool):
    """运行系统健康检查（配置、依赖、数据库连接等）。

    \b
    示例:
      deepmt health check
      deepmt health check --deep
    """
    try:
        from deepmt.core.health_checker import HealthChecker
        checker = HealthChecker()
        if deep:
            report = checker.run_deep_checks()
            checker.print_report(report)
        else:
            checker.print_report()
    except Exception as e:
        click.echo(click.style(f"健康检查失败: {e}", fg="red"), err=True)
        sys.exit(1)


@health.command("matrix")
@click.option("--json", "as_json", is_flag=True, help="输出 JSON 格式")
def health_matrix(as_json: bool):
    """输出算子 × 框架 可达性矩阵。

    \b
    示例:
      deepmt health matrix
      deepmt health matrix --json
    """
    try:
        from deepmt.core.health_checker import HealthChecker
        checker = HealthChecker()
        matrix = checker.compute_reachability_matrix()
        if as_json:
            import json
            click.echo(json.dumps(matrix, indent=2, ensure_ascii=False))
            return
        if not matrix:
            click.echo("（知识库为空或无可用插件）")
            return
        frameworks = sorted({f for row in matrix.values() for f in row})
        header = "operator".ljust(20) + "".join(f.ljust(16) for f in frameworks)
        click.echo(header)
        click.echo("─" * len(header))
        for op in sorted(matrix):
            row = matrix[op]
            cells = []
            for f in frameworks:
                mark = "✅" if row.get(f) else "·"
                cells.append(f"  {mark}".ljust(16))
            click.echo(op.ljust(20) + "".join(cells))
    except Exception as e:
        click.echo(click.style(f"matrix 生成失败: {e}", fg="red"), err=True)
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
