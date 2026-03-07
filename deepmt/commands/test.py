"""
deepmt test — 测试执行子命令组

命令:
    operator      测试单个算子
    from-config   从 YAML 配置文件批量测试
    history       查看测试历史
    failures      查看失败的测试用例
"""

import json
import sys

import click

from deepmt._utils import get_results_manager, not_implemented_error

_SUPPORTED_FRAMEWORKS = {"pytorch"}
_ALL_FRAMEWORKS = {"pytorch", "tensorflow", "paddlepaddle"}


def _check_framework(framework: str):
    if framework not in _SUPPORTED_FRAMEWORKS:
        not_implemented_error(
            f"--framework {framework}",
            f"框架 '{framework}' 的插件尚未实现，目前仅支持 pytorch。",
        )


@click.group()
def test():
    """测试执行（运行算子 / 从配置文件 / 查看历史）。"""


# ── operator ──────────────────────────────────────────────────────────────────

@test.command("operator")
@click.argument("operator")
@click.option(
    "--framework",
    default="pytorch",
    type=click.Choice(list(_ALL_FRAMEWORKS), case_sensitive=False),
    show_default=True,
    help="目标框架",
)
@click.option(
    "--inputs",
    default="1.0,2.0",
    show_default=True,
    help="输入值，逗号分隔的浮点数",
)
@click.option("--generate/--no-generate", default=True, show_default=True, help="若知识库无 MR 则自动生成")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出结果")
def test_operator(operator, framework, inputs, generate, as_json):
    """对算子运行蜕变测试。

    若知识库中已有该算子的 MR，则直接加载；否则自动生成（需要 LLM 配置）。

    \b
    示例:
      deepmt test operator relu
      deepmt test operator relu --framework pytorch --inputs 1.0,2.0,-1.0
      deepmt test operator add --inputs 1.0,2.0 --json
    """
    _check_framework(framework)

    try:
        input_list = [float(x.strip()) for x in inputs.split(",")]
    except ValueError:
        raise click.BadParameter(f"--inputs 格式错误，期望逗号分隔的浮点数，如 '1.0,2.0'")

    click.echo(f"[test] 算子: {operator}  框架: {framework}  inputs: {input_list}")

    try:
        from api.deepmt import DeepMT

        client = DeepMT()
        result = client.test_operator(
            name=operator,
            inputs=input_list,
            framework=framework,
        )

        if as_json:
            click.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            click.echo(result.summary())

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)


# ── from-config ───────────────────────────────────────────────────────────────

@test.command("from-config")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出结果")
def test_from_config(config_path, as_json):
    """从 YAML 配置文件批量运行测试。

    配置文件格式示例:

    \b
      tests:
        - type: operator
          name: relu
          inputs: [1.0, -1.0, 0.0]
          framework: pytorch
        - type: operator
          name: add
          inputs: [1.0, 2.0]
          framework: pytorch

    \b
    示例:
      deepmt test from-config tests/my_tests.yaml
      deepmt test from-config tests/my_tests.yaml --json
    """
    click.echo(f"[from-config] 配置文件: {config_path}")

    try:
        from api.deepmt import DeepMT

        client = DeepMT()
        results = client.test_from_config(config_path)

        if as_json:
            click.echo(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        else:
            click.echo(f"\n共执行 {len(results)} 个测试:")
            for r in results:
                status = click.style("PASS", fg="green") if r.failed_tests == 0 else click.style("FAIL", fg="red")
                click.echo(f"  [{status}] {r.name}  passed={r.passed_tests}/{r.total_tests}")

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)


# ── history ───────────────────────────────────────────────────────────────────

@test.command("history")
@click.argument("name", required=False, default=None)
@click.option("--limit", default=20, show_default=True, type=int, help="最多显示条数")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def test_history(name, limit, as_json):
    """查看测试历史。

    \b
    示例:
      deepmt test history
      deepmt test history relu
      deepmt test history --limit 50 --json
    """
    try:
        rm = get_results_manager()
        records = rm.get_summary(name)

        if limit:
            records = records[:limit]

        if as_json:
            click.echo(json.dumps(records, ensure_ascii=False, indent=2))
            return

        if not records:
            click.echo("暂无测试历史记录。")
            return

        click.echo(f"测试历史（共 {len(records)} 条）:")
        click.echo("─" * 70)
        for r in records:
            total = r.get("total_tests", 0)
            passed = r.get("passed_tests", 0)
            failed = r.get("failed_tests", 0)
            ir_name = r.get("ir_name", "?")
            status_str = click.style("PASS", fg="green") if failed == 0 else click.style("FAIL", fg="red")
            click.echo(f"  [{status_str}] {ir_name}  passed={passed}/{total}")

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)


# ── failures ──────────────────────────────────────────────────────────────────

@test.command("failures")
@click.option("--limit", default=50, show_default=True, type=int, help="最多显示条数")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def test_failures(limit, as_json):
    """查看失败的测试用例。

    \b
    示例:
      deepmt test failures
      deepmt test failures --limit 100
      deepmt test failures --json
    """
    try:
        rm = get_results_manager()
        records = rm.get_failed_tests(limit)

        if as_json:
            click.echo(json.dumps(records, ensure_ascii=False, indent=2))
            return

        if not records:
            click.echo(click.style("未发现失败测试用例。", fg="green"))
            return

        click.echo(f"失败测试用例（共 {len(records)} 条）:")
        click.echo("─" * 70)
        for r in records:
            click.echo(f"  {r.get('ir_name', '?')} / MR: {r.get('mr_description', '?')}")
            if r.get("defect_type"):
                click.echo(f"    缺陷类型: {r['defect_type']}")

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)
