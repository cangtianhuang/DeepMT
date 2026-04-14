"""
deepmt.commands.test.execution — 测试执行子命令

命令: operator, batch, model, cross, from-config
"""

import json
import sys

import click

from deepmt.commands.test._group import test, _check_framework, _ALL_FRAMEWORKS


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
@click.option("--collect-evidence", is_flag=True, default=False, help="失败时捕获可复现证据包并保存到 data/results/evidence/")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出结果")
def test_batch(framework, operator, category, mr_id, n_samples, verified_only, collect_evidence, as_json):
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
                    collect_evidence=collect_evidence,
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
            if s.evidence_ids:
                click.echo(f"      证据包: {', '.join(s.evidence_ids)}")

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


# ── cross ─────────────────────────────────────────────────────────────────────

_CROSS_FRAMEWORKS = ["pytorch", "numpy"]


@test.command("cross")
@click.argument("operator")
@click.option(
    "--framework1",
    default="pytorch",
    show_default=True,
    help="第一框架（主框架）",
)
@click.option(
    "--framework2",
    default="numpy",
    show_default=True,
    help="第二框架（参考框架；可选 numpy/paddlepaddle/paddle，默认 numpy）",
)
@click.option("--n-samples", default=20, show_default=True, type=int, help="每条 MR 的测试样本数")
@click.option("--verified-only", is_flag=True, default=False, help="仅使用已验证的 MR")
@click.option("--save", is_flag=True, default=False, help="将结果保存到 data/results/cross_framework/")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def test_cross(operator, framework1, framework2, n_samples, verified_only, save, as_json):
    """跨框架一致性测试：对比两框架在等价算子上的 MR 结论是否一致。

    默认以 pytorch 为主框架，numpy 为参考框架。
    一致性指：两框架对同一输入同一 MR，结论相同（both pass 或 both fail）。

    \b
    示例:
      deepmt test cross torch.nn.functional.relu
      deepmt test cross torch.exp --n-samples 30 --save
      deepmt test cross torch.tanh --framework1 pytorch --framework2 numpy --json
      deepmt test cross torch.abs --framework2 paddlepaddle --save
      deepmt test cross torch.nn.functional.relu --framework2 paddle --json
    """
    try:
        from deepmt.analysis.qa.cross_framework_tester import CrossFrameworkTester

        tester = CrossFrameworkTester()
        session = tester.compare_operator(
            operator_name=operator,
            framework1=framework1,
            framework2=framework2,
            n_samples=n_samples,
            verified_only=verified_only,
        )

        if save:
            path = tester.save(session)
            click.echo(f"已保存: {path}")

        if as_json:
            click.echo(json.dumps(session.to_dict(), ensure_ascii=False, indent=2))
            return

        click.echo(f"\n跨框架一致性测试 — {operator}")
        click.echo(f"  {framework1} vs {framework2}  |  n_samples={n_samples}")
        click.echo("─" * 72)
        click.echo(
            f"  MR 数: {session.mr_count}"
            f"  整体一致率: {session.overall_consistency_rate:.1%}"
            f"  输出最大差: {session.output_max_diff:.4g}"
        )
        click.echo("─" * 72)
        for r in session.mr_results:
            mark = (
                click.style("≈", fg="green") if r.consistency_rate >= 0.9
                else click.style("!", fg="yellow") if r.inconsistent_cases > 0
                else click.style("≠", fg="red")
            )
            click.echo(
                f"  [{mark}] {r.mr_description[:50]}"
                f"  consistency={r.consistency_rate:.0%}"
                f"  f1_pass={r.f1_pass_rate:.0%}"
                f"  f2_pass={r.f2_pass_rate:.0%}"
                f"  diff={r.output_max_diff:.3g}"
            )
            if r.inconsistent_cases > 0:
                click.echo(
                    f"      不一致样本: {r.inconsistent_cases}/{r.total_valid}"
                    f"  (仅f1通过={r.only_f1_pass}, 仅f2通过={r.only_f2_pass})"
                )
            if r.diff_type_counts:
                diff_str = "  ".join(f"{k}={v}" for k, v in r.diff_type_counts.items() if v > 0)
                click.echo(f"      差异类型: {diff_str}")
        click.echo("─" * 72)
        click.echo(
            f"  ≈ 高度一致（≥90%）  ! 存在不一致  ≠ 低一致率"
        )

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
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


# ── model ──────────────────────────────────────────────────────────────────────

@test.command("model")
@click.argument("model_name", required=False, default=None)
@click.option(
    "--framework",
    default="pytorch",
    type=click.Choice(["pytorch"], case_sensitive=False),
    show_default=True,
    help="目标框架（当前仅支持 pytorch）",
)
@click.option("--n-samples", default=10, show_default=True, type=int, help="每条 MR 的测试样本数")
@click.option("--max-mrs", default=None, type=int, help="每个模型最多使用的 MR 数量")
@click.option("--batch-size", default=4, show_default=True, type=int, help="每次推理的 batch 大小")
@click.option("--all", "run_all", is_flag=True, default=False, help="测试所有基准模型")
@click.option("--list", "list_models", is_flag=True, default=False, help="列出可用的基准模型")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出结果")
def test_model(model_name, framework, n_samples, max_mrs, batch_size, run_all, list_models, as_json):
    """模型层蜕变测试：对基准模型自动生成并执行 MR 测试。

    基于结构分析自动生成模型层蜕变关系，无需 LLM。
    支持 MLP、CNN、RNN、Transformer 四类基准模型。

    \b
    示例:
      deepmt test model --list                          # 列出可用基准模型
      deepmt test model SimpleMLP                       # 测试 SimpleMLP
      deepmt test model SimpleCNN --n-samples 20
      deepmt test model --all                           # 测试所有基准模型
      deepmt test model SimpleMLP --json
    """
    try:
        from deepmt.benchmarks.models import ModelBenchmarkRegistry
        from deepmt.engine.model_test_runner import ModelTestRunner

        registry = ModelBenchmarkRegistry()

        if list_models:
            names = registry.names(framework=framework)
            click.echo(f"可用基准模型（框架={framework}，共 {len(names)} 个）:")
            for n in names:
                ir = registry.get(n, with_instance=False)
                click.echo(
                    f"  {n:<20} type={ir.model_type:<12} "
                    f"input={ir.input_shape}  classes={ir.num_classes}"
                )
            return

        runner = ModelTestRunner()

        if run_all:
            summaries = runner.run_all(
                framework=framework,
                n_samples=n_samples,
                max_mrs=max_mrs,
                batch_size=batch_size,
            )
            if as_json:
                click.echo(json.dumps(
                    [s.to_dict() for s in summaries], ensure_ascii=False, indent=2
                ))
                return
            total_cases = sum(s.total_cases for s in summaries)
            total_passed = sum(s.passed for s in summaries)
            click.echo(f"\n模型层批量测试结果（{len(summaries)} 个模型）")
            click.echo("─" * 60)
            for s in summaries:
                status = click.style("PASS", fg="green") if s.failed == 0 else click.style("FAIL", fg="red")
                click.echo(
                    f"  [{status}] {s.model_name:<20} "
                    f"MRs={s.mr_count}  "
                    f"{s.passed}/{s.total_cases} ({s.pass_rate:.0%})"
                )
            click.echo("─" * 60)
            overall_rate = total_passed / total_cases if total_cases > 0 else 0.0
            click.echo(
                f"  汇总: {total_passed}/{total_cases} passed ({overall_rate:.1%})"
            )
            return

        if model_name is None:
            click.echo(
                click.style(
                    "请指定模型名称（如 SimpleMLP），或使用 --all 测试所有模型，"
                    "或使用 --list 列出可用模型。",
                    fg="yellow",
                ),
                err=True,
            )
            sys.exit(1)

        summary = runner.run_model(
            model_name,
            n_samples=n_samples,
            max_mrs=max_mrs,
            batch_size=batch_size,
        )

        if as_json:
            click.echo(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
            return

        status_str = click.style("PASS", fg="green") if summary.failed == 0 else click.style("FAIL", fg="red")
        click.echo(f"\n模型层测试结果 [{status_str}]")
        click.echo("─" * 60)
        click.echo(f"  模型:    {summary.model_name}  ({summary.model_type})")
        click.echo(f"  框架:    {summary.framework}")
        click.echo(f"  MR 数:   {summary.mr_count}")
        click.echo(f"  样本数:  {summary.n_samples}")
        click.echo(f"  结果:    {summary.passed}/{summary.total_cases} passed  "
                   f"errors={summary.errors}  pass_rate={summary.pass_rate:.1%}")
        if summary.mr_summaries:
            click.echo(f"\n  逐 MR 统计:")
            for m in summary.mr_summaries:
                flag = "✓" if m.failed == 0 else "✗"
                click.echo(
                    f"    [{flag}] {m.description[:55]:<55}  "
                    f"{m.passed}/{m.total} ({m.pass_rate:.0%})"
                )
        if summary.failure_cases:
            click.echo(f"\n  失败案例（前 {len(summary.failure_cases)} 个）:")
            for fc in summary.failure_cases[:5]:
                click.echo(f"    sample#{fc['sample_idx']+1}: {fc['detail'][:80]}")

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)
