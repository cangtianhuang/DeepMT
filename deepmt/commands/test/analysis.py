"""
deepmt.commands.test.analysis — 分析类测试子命令

命令: mutate, open, report, dedup
"""

import json
import sys

import click

from deepmt.commands.test._group import test, _check_framework, _ALL_FRAMEWORKS, _MUTANT_TYPES


# ── mutate ────────────────────────────────────────────────────────────────────

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
        from deepmt.analysis.reporting.mutation_tester import MutantType, MutationTester

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


# ── open ─────────────────────────────────────────────────────────────────────

@test.command("open")
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
    help="指定单个算子（不指定则测试知识库中所有算子）",
)
@click.option(
    "--inject-faults",
    "inject_faults",
    default=None,
    help=(
        "缺陷注入规格，优先于 DEEPMT_INJECT_FAULTS 环境变量。"
        " 格式: 'all' 或 'op1:mutant1,op2:mutant2'。"
        " 例: 'torch.nn.functional.relu:negate,torch.exp:scale'"
    ),
)
@click.option(
    "--list-catalog",
    "list_catalog",
    is_flag=True,
    default=False,
    help="列出内置缺陷目录后退出",
)
@click.option("--n-samples", default=10, show_default=True, type=int, help="每条 MR 的随机测试样本数")
@click.option("--verified-only", is_flag=True, default=False, help="仅使用已验证的 MR")
@click.option("--collect-evidence", is_flag=True, default=False, help="失败时保存可复现证据包")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出结果")
def test_open(framework, operator, inject_faults, list_catalog, n_samples, verified_only, collect_evidence, as_json):
    """开放测试：对含预设缺陷的插件运行批量蜕变测试（受控真实场景）。

    使用 FaultyPyTorchPlugin 代替正常 PyTorchPlugin，将指定算子替换为含已知缺陷的版本，
    验证 DeepMT 能否在"真实框架缺陷"场景中检测问题。

    缺陷来源（优先级从高到低）：
      1. --inject-faults 命令行参数
      2. DEEPMT_INJECT_FAULTS 环境变量
      3. 若均未设置，使用完整内置缺陷目录

    \b
    示例:
      deepmt test open --list-catalog
      deepmt test open --operator torch.nn.functional.relu --inject-faults all
      deepmt test open --inject-faults torch.exp:scale --n-samples 20 --collect-evidence
      DEEPMT_INJECT_FAULTS=all deepmt test open
    """
    _check_framework(framework)

    try:
        from deepmt.plugins.faulty_pytorch_plugin import (
            BUILTIN_FAULT_CATALOG,
            FaultyPyTorchPlugin,
        )

        if list_catalog:
            click.echo("\n内置缺陷目录：")
            click.echo("─" * 72)
            for op, (mt, desc) in FaultyPyTorchPlugin.list_catalog().items():
                click.echo(f"  {op}")
                click.echo(f"    变异类型: {mt}")
                click.echo(f"    描述:     {desc}")
                click.echo()
            return

        import os

        if inject_faults:
            os.environ["DEEPMT_INJECT_FAULTS"] = inject_faults
            faulty_plugin = FaultyPyTorchPlugin()
            del os.environ["DEEPMT_INJECT_FAULTS"]
        else:
            env_val = os.environ.get("DEEPMT_INJECT_FAULTS", "")
            if not env_val:
                click.echo(
                    click.style(
                        "提示: 未指定 --inject-faults 且 DEEPMT_INJECT_FAULTS 未设置，"
                        "将使用完整内置缺陷目录（all）。",
                        fg="yellow",
                    )
                )
                faulty_plugin = FaultyPyTorchPlugin(
                    fault_specs={op: mt for op, (mt, _, _) in BUILTIN_FAULT_CATALOG.items()}
                )
            else:
                faulty_plugin = FaultyPyTorchPlugin()

        active = faulty_plugin.active_faults_from_env() if not inject_faults else {}
        active_count = len(faulty_plugin._active_faults)

        click.echo(f"\n开放测试（framework={framework}）| 激活缺陷算子: {active_count} 个")
        click.echo("─" * 72)
        for op, (mt, _, desc) in faulty_plugin._active_faults.items():
            click.echo(f"  ⚠  {op}  [{mt}]  {desc[:60]}")
        click.echo("─" * 72)

        from deepmt.engine.batch_test_runner import BatchTestRunner

        runner = BatchTestRunner(
            backend_override=faulty_plugin,
            evidence_collector=None,
        )

        if operator:
            summaries = [
                runner.run_operator(
                    operator_name=operator,
                    framework=framework,
                    n_samples=n_samples,
                    verified_only=verified_only,
                    collect_evidence=collect_evidence,
                )
            ]
        else:
            summaries = runner.run_batch(
                framework=framework,
                n_samples=n_samples,
                verified_only=verified_only,
            )

        if as_json:
            click.echo(json.dumps([s.to_dict() for s in summaries], ensure_ascii=False, indent=2))
            return

        if not summaries:
            click.echo("没有找到可测试的算子（MR 知识库为空）。")
            return

        detected_ops = [s for s in summaries if s.failed > 0]
        click.echo(f"\n开放测试结果（framework={framework}）")
        click.echo("─" * 72)
        for s in summaries:
            if s.mr_count == 0:
                continue
            status_str = (
                click.style("检出", fg="green") if s.failed > 0
                else click.style("漏检", fg="red")
            )
            click.echo(
                f"  [{status_str}] {s.operator}"
                f"  fail={s.failed}/{s.total_cases}"
                f"  err={s.errors}"
            )
            if s.evidence_ids:
                click.echo(f"      证据: {', '.join(s.evidence_ids)}")
        click.echo("─" * 72)
        click.echo(
            f"算子检出: {len(detected_ops)}/{len([s for s in summaries if s.mr_count > 0])}  |  "
            f"总失败: {sum(s.failed for s in summaries)}"
        )

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
        from deepmt.analysis.reporting.report_generator import ReportGenerator

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


# ── dedup ─────────────────────────────────────────────────────────────────────

@test.command("dedup")
@click.option("--operator", default=None, help="按算子名称过滤")
@click.option("--framework", default=None, help="按框架过滤")
@click.option("--limit", default=0, show_default=True, type=int, help="最多显示条数（0=不限）")
@click.option(
    "--source",
    default="evidence",
    type=click.Choice(["evidence", "cross", "all"], case_sensitive=False),
    show_default=True,
    help="聚类输入来源：evidence=证据包；cross=跨框架会话；all=两者合并",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def test_dedup(operator, framework, limit, source, as_json):
    """缺陷线索去重：将失败证据包聚类为独立缺陷模式。

    从 data/results/evidence/ 读取已保存的证据包，按（算子 × MR × 错误类型）签名聚类，
    将大量重复失败压缩为可人工复核的缺陷线索集。

    前提：先运行 deepmt test batch --collect-evidence 或 deepmt test open --collect-evidence
    收集证据包。

    \b
    示例:
      deepmt test dedup
      deepmt test dedup --operator torch.nn.functional.relu
      deepmt test dedup --limit 10 --json
    """
    try:
        from deepmt.analysis.qa.defect_deduplicator import DefectDeduplicator

        dedup = DefectDeduplicator()
        leads = dedup.deduplicate(
            operator=operator, framework=framework, limit=limit, source=source.lower(),
        )

        if as_json:
            click.echo(json.dumps([l.to_dict() for l in leads], ensure_ascii=False, indent=2))
            return

        click.echo(dedup.format_text(leads))

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)
