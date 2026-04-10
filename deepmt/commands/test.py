"""
deepmt test — 测试执行子命令组

命令:
    operator      测试单个算子
    batch         批量测试（从 MR 知识库自动选取算子，RandomGenerator 生成输入）
    mutate        变异测试（注入已知错误实现，验证缺陷检测能力）
    report        生成测试结果报告
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
        from deepmt.client import DeepMT

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


# ── batch ─────────────────────────────────────────────────────────────────────

@test.command("batch")
@click.option(
    "--framework",
    default="pytorch",
    type=click.Choice(list(_ALL_FRAMEWORKS), case_sensitive=False),
    show_default=True,
    help="目标框架",
)
@click.option(
    "--operator",
    default=None,
    help="指定单个算子名称（如 torch.nn.functional.relu）；不指定则测试知识库中所有算子",
)
@click.option(
    "--category",
    default=None,
    help="按算子目录分类过滤（如 activation）",
)
@click.option(
    "--mr-id",
    default=None,
    help="指定单条 MR ID；不指定则测试算子的全部 MR",
)
@click.option(
    "--n-samples",
    default=10,
    show_default=True,
    type=int,
    help="每条 MR 的随机测试样本数",
)
@click.option(
    "--verified-only",
    is_flag=True,
    default=False,
    help="仅使用已通过验证（verified=True）的 MR",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出结果")
def test_batch(framework, operator, category, mr_id, n_samples, verified_only, as_json):
    """从 MR 知识库批量执行蜕变测试（自动生成输入）。

    系统从 MR 知识库中读取已生成的 MR，调用 RandomGenerator 自动生成随机输入，
    批量执行测试并将结果存储到数据库。

    \b
    示例:
      deepmt test batch                                    # 测试所有算子
      deepmt test batch --operator torch.nn.functional.relu
      deepmt test batch --framework pytorch --n-samples 20
      deepmt test batch --category activation --verified-only
      deepmt test batch --operator torch.exp --json
    """
    _check_framework(framework)

    try:
        from deepmt.engine.batch_test_runner import BatchTestRunner

        runner = BatchTestRunner()

        if operator:
            summaries = [
                runner.run_operator(
                    operator_name=operator,
                    framework=framework,
                    n_samples=n_samples,
                    verified_only=verified_only,
                    mr_id=mr_id,
                )
            ]
        else:
            summaries = runner.run_batch(
                framework=framework,
                category=category,
                n_samples=n_samples,
                verified_only=verified_only,
            )

        if as_json:
            click.echo(json.dumps([s.to_dict() for s in summaries], ensure_ascii=False, indent=2))
            return

        if not summaries:
            click.echo("没有找到可测试的算子（MR 知识库为空或过滤条件无匹配）。")
            return

        click.echo(f"\n批量测试结果（framework={framework}）")
        click.echo("─" * 70)

        for s in summaries:
            if s.mr_count == 0:
                status_str = click.style("SKIP", fg="yellow")
                click.echo(f"  [{status_str}] {s.operator}  (无可用 MR)")
                continue

            status_str = (
                click.style("PASS", fg="green")
                if s.failed == 0 and s.errors == 0
                else click.style("FAIL", fg="red")
            )
            click.echo(
                f"  [{status_str}] {s.operator}"
                f"  MR={s.mr_count}"
                f"  samples={s.n_samples}"
                f"  pass={s.passed}/{s.total_cases}"
                f"  err={s.errors}"
            )
            for m in s.mr_summaries:
                mr_status = click.style("✓", fg="green") if m.failed == 0 and m.errors == 0 else click.style("✗", fg="red")
                click.echo(
                    f"      {mr_status} {m.description[:55]}"
                    f"  pass={m.passed}/{m.total}"
                    + (f"  err={m.errors}" if m.errors else "")
                )

        click.echo("─" * 70)
        total_ops = len(summaries)
        tested_ops = sum(1 for s in summaries if s.mr_count > 0)
        total_cases = sum(s.total_cases for s in summaries)
        total_passed = sum(s.passed for s in summaries)
        click.echo(
            f"算子: {tested_ops}/{total_ops}  |  "
            f"总用例: {total_cases}  |  "
            f"通过: {total_passed}  |  "
            f"失败: {total_cases - total_passed}"
        )

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)


# ── mutate ────────────────────────────────────────────────────────────────────

_MUTANT_TYPES = ["negate", "add_const", "scale", "identity", "zero"]


@test.command("mutate")
@click.argument("operator")
@click.option(
    "--framework",
    default="pytorch",
    type=click.Choice(list(_ALL_FRAMEWORKS), case_sensitive=False),
    show_default=True,
    help="目标框架",
)
@click.option(
    "--mutant",
    "mutant_type",
    default=None,
    type=click.Choice(_MUTANT_TYPES, case_sensitive=False),
    help="变异类型；不指定则运行全部变异类型",
)
@click.option("--n-samples", default=10, show_default=True, type=int, help="每条 MR 的测试样本数")
@click.option("--verified-only", is_flag=True, default=False, help="仅使用已验证的 MR")
@click.option("--scale", default=2.0, show_default=True, type=float, help="scale 变异的缩放系数")
@click.option("--const", default=1.0, show_default=True, type=float, help="add_const 变异的偏置值")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出结果")
def test_mutate(operator, framework, mutant_type, n_samples, verified_only, scale, const, as_json):
    """对算子注入已知错误实现，验证 MR 的缺陷检测能力。

    变异测试用于受控评估：若系统无法发现已知缺陷，说明 MR 或执行链路存在问题。

    \b
    变异类型:
      negate    — 取反输出: f(x) = -real_f(x)
      add_const — 添加偏置: f(x) = real_f(x) + C
      scale     — 错误缩放: f(x) = k * real_f(x)
      identity  — 恒等函数: f(x) = x
      zero      — 恒零输出: f(x) = 0

    \b
    示例:
      deepmt test mutate torch.nn.functional.relu
      deepmt test mutate torch.nn.functional.relu --mutant negate
      deepmt test mutate torch.exp --mutant add_const --const 100 --json
    """
    _check_framework(framework)

    try:
        from deepmt.analysis.mutation_tester import MutantType, MutationTester

        tester = MutationTester()

        if mutant_type:
            results = [
                tester.run(
                    operator_name=operator,
                    mutant_type=MutantType(mutant_type),
                    framework=framework,
                    n_samples=n_samples,
                    verified_only=verified_only,
                    scale=scale,
                    const=const,
                )
            ]
        else:
            results = tester.run_all_mutants(
                operator_name=operator,
                framework=framework,
                n_samples=n_samples,
                verified_only=verified_only,
            )

        if as_json:
            click.echo(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
            return

        click.echo(f"\n变异测试结果 — {operator} (framework={framework})")
        click.echo("─" * 70)
        for r in results:
            detected_str = (
                click.style("检出", fg="green") if r.detected else click.style("未检出", fg="red")
            )
            click.echo(
                f"  [{detected_str}] {str(r.mutant_type):<12}"
                f"  检出={r.detected_cases}/{r.total_cases}"
                f"  检出率={r.detection_rate:.1%}"
                f"  err={r.errors}"
            )
            for m in r.mr_details:
                mr_mark = "✓" if m["detected_cases"] > 0 else "·"
                desc = m.get("description", m.get("mr_id", "?"))[:50]
                click.echo(f"      {mr_mark} {desc:<50}  {m['detected_cases']}/{m['total_cases']}")
        click.echo("─" * 70)

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)


# ── report ────────────────────────────────────────────────────────────────────


@test.command("report")
@click.option(
    "--framework",
    default=None,
    type=click.Choice(list(_ALL_FRAMEWORKS) + ["all"], case_sensitive=False),
    help="按框架过滤（不指定则显示全部）",
)
@click.option("--operator", default=None, help="按算子名称过滤")
@click.option("--failures-only", is_flag=True, default=False, help="仅显示失败案例")
@click.option("--limit", default=0, show_default=True, type=int, help="最多显示算子数（0=不限）")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def test_report(framework, operator, failures_only, limit, as_json):
    """生成测试结果报告。

    从数据库读取历史测试记录，汇总通过率、失败分布、逐 MR 明细。

    \b
    示例:
      deepmt test report
      deepmt test report --framework pytorch
      deepmt test report --operator torch.nn.functional.relu
      deepmt test report --failures-only
      deepmt test report --json
    """
    fw = None if framework == "all" else framework

    try:
        from deepmt.analysis.report_generator import ReportGenerator

        gen = ReportGenerator()

        if failures_only:
            failures = gen.get_failures(framework=fw, operator=operator, limit=limit or 50)
            if as_json:
                click.echo(json.dumps(failures, ensure_ascii=False, indent=2))
            else:
                click.echo(gen.format_failure_text(failures))
            return

        report = gen.generate(framework=fw, operator=operator, limit=limit)

        if as_json:
            click.echo(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            click.echo(gen.format_text(report))

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
        from deepmt.client import DeepMT

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
