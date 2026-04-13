"""
deepmt test — 测试执行子命令组

命令:
    operator      测试单个算子
    batch         批量测试（从 MR 知识库自动选取算子，RandomGenerator 生成输入）
    mutate        变异测试（注入已知错误实现，验证缺陷检测能力）
    open          开放测试（对含预设缺陷的插件运行批量测试，模拟真实框架缺陷场景）
    report        生成测试结果报告
    dedup         缺陷线索去重（将失败证据包聚类为独立缺陷线索）
    evidence      证据包管理（list / show / script）
    cross         跨框架一致性测试（D6，对比两个框架在等价算子上的行为）
    experiment    论文实验数据汇总（D7，RQ1-RQ4 数据组织）
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

        # 解析 fault_specs
        import os

        if inject_faults:
            # 临时写入环境变量（仅本进程，不污染用户环境）
            os.environ["DEEPMT_INJECT_FAULTS"] = inject_faults
            faulty_plugin = FaultyPyTorchPlugin()
            del os.environ["DEEPMT_INJECT_FAULTS"]
        else:
            env_val = os.environ.get("DEEPMT_INJECT_FAULTS", "")
            if not env_val:
                # 默认：激活全部内置缺陷
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
            evidence_collector=None,  # 使用默认（若 collect_evidence 则自动创建）
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


# ── dedup ─────────────────────────────────────────────────────────────────────


@test.command("dedup")
@click.option("--operator", default=None, help="按算子名称过滤")
@click.option("--framework", default=None, help="按框架过滤")
@click.option("--limit", default=0, show_default=True, type=int, help="最多显示条数（0=不限）")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def test_dedup(operator, framework, limit, as_json):
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
        from deepmt.analysis.defect_deduplicator import DefectDeduplicator

        dedup = DefectDeduplicator()
        leads = dedup.deduplicate(operator=operator, framework=framework, limit=limit)

        if as_json:
            click.echo(json.dumps([l.to_dict() for l in leads], ensure_ascii=False, indent=2))
            return

        click.echo(dedup.format_text(leads))

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)


# ── evidence ──────────────────────────────────────────────────────────────────


@test.group("evidence")
def test_evidence():
    """证据包管理：查看、展示可复现失败案例。"""


@test_evidence.command("list")
@click.option("--operator", default=None, help="按算子名称过滤")
@click.option("--framework", default=None, help="按框架过滤")
@click.option("--limit", default=20, show_default=True, type=int, help="最多显示条数（0=不限）")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def evidence_list(operator, framework, limit, as_json):
    """列出已保存的证据包。

    \b
    示例:
      deepmt test evidence list
      deepmt test evidence list --operator torch.nn.functional.relu
      deepmt test evidence list --limit 5 --json
    """
    try:
        from deepmt.analysis.evidence_collector import EvidenceCollector

        collector = EvidenceCollector()
        packs = collector.list_all(operator=operator, framework=framework, limit=limit)

        if as_json:
            click.echo(json.dumps([p.to_dict() for p in packs], ensure_ascii=False, indent=2))
            return

        if not packs:
            click.echo("暂无证据包记录。运行 deepmt test batch 时添加 --collect-evidence 可生成证据包。")
            return

        click.echo(f"\n证据包列表（共 {len(packs)} 条）")
        click.echo("─" * 70)
        for p in packs:
            click.echo(
                f"  {p.evidence_id}  {p.timestamp[:16]}"
                f"  {p.operator}  [{p.framework} {p.framework_version}]"
            )
            click.echo(f"    MR: {p.mr_description[:60]}")
            click.echo(f"    diff={p.actual_diff:.4g}  tol={p.tolerance:.4g}  {p.detail[:50]}")
        click.echo("─" * 70)

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)


@test_evidence.command("show")
@click.argument("evidence_id")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出完整数据")
def evidence_show(evidence_id, as_json):
    """显示单个证据包的详细信息。

    \b
    示例:
      deepmt test evidence show abc123def456
      deepmt test evidence show abc123def456 --json
    """
    try:
        from deepmt.analysis.evidence_collector import EvidenceCollector

        collector = EvidenceCollector()
        pack = collector.load(evidence_id)

        if pack is None:
            click.echo(click.style(f"未找到证据包 {evidence_id!r}", fg="red"), err=True)
            sys.exit(1)

        if as_json:
            click.echo(json.dumps(pack.to_dict(), ensure_ascii=False, indent=2))
            return

        click.echo(f"\n证据包详情 — {pack.evidence_id}")
        click.echo("─" * 70)
        click.echo(f"  时间:     {pack.timestamp}")
        click.echo(f"  算子:     {pack.operator}")
        click.echo(f"  框架:     {pack.framework} {pack.framework_version}")
        click.echo(f"  MR ID:    {pack.mr_id}")
        click.echo(f"  MR 描述:  {pack.mr_description}")
        click.echo(f"  变换代码: {pack.transform_code}")
        click.echo(f"  Oracle:   {pack.oracle_expr}")
        click.echo(f"  实测差值: {pack.actual_diff:.6g}  (容忍阈值: {pack.tolerance:.6g})")
        click.echo(f"  失败原因: {pack.detail}")
        shape = pack.input_summary.get("shape", "?")
        dtype = pack.input_summary.get("dtype", "?")
        click.echo(f"  输入形状: {shape}  dtype={dtype}")
        click.echo("─" * 70)

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)


@test_evidence.command("script")
@click.argument("evidence_id")
def evidence_script(evidence_id):
    """打印指定证据包的可复现 Python 脚本。

    \b
    示例:
      deepmt test evidence script abc123def456
      deepmt test evidence script abc123def456 > repro.py
    """
    try:
        from deepmt.analysis.evidence_collector import EvidenceCollector

        collector = EvidenceCollector()
        pack = collector.load(evidence_id)

        if pack is None:
            click.echo(click.style(f"未找到证据包 {evidence_id!r}", fg="red"), err=True)
            sys.exit(1)

        click.echo(pack.reproduce_script)

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
        from deepmt.analysis.cross_framework_tester import CrossFrameworkTester

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

        # 文本输出
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


# ── experiment ────────────────────────────────────────────────────────────────


@test.command("experiment")
@click.option(
    "--rq",
    default="all",
    type=click.Choice(["1", "2", "3", "4", "all"], case_sensitive=False),
    show_default=True,
    help="收集指定 RQ 的数据（1/2/3/4/all）",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def test_experiment(rq, as_json):
    """论文实验数据汇总：将系统产出映射到 RQ1-RQ4。

    \b
    RQ1 — MR 生成质量（总数、验证率、分类分布）
    RQ2 — 缺陷检测能力（通过率、失败分布、证据包数）
    RQ3 — 跨框架一致性（一致率、输出差、不一致案例）
    RQ4 — 覆盖度与自动化程度

    \b
    示例:
      deepmt test experiment
      deepmt test experiment --rq 2
      deepmt test experiment --json > experiment_data.json
    """
    try:
        from deepmt.analysis.experiment_organizer import ExperimentOrganizer

        org = ExperimentOrganizer()

        if rq == "all":
            data = org.collect_all()
            if as_json:
                click.echo(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                click.echo(org.format_text(data))
        else:
            fn = {
                "1": org.collect_rq1,
                "2": org.collect_rq2,
                "3": org.collect_rq3,
                "4": org.collect_rq4,
            }[rq]
            result = fn()
            if as_json:
                click.echo(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                title = {
                    "1": "RQ1 — MR 生成质量",
                    "2": "RQ2 — 缺陷检测能力",
                    "3": "RQ3 — 跨框架一致性",
                    "4": "RQ4 — 覆盖度与自动化",
                }[rq]
                click.echo(f"\n{title}")
                click.echo("─" * 60)
                for k, v in result.items():
                    if isinstance(v, dict):
                        click.echo(f"  {k}:")
                        for kk, vv in v.items():
                            click.echo(f"    {kk}: {vv}")
                    else:
                        click.echo(f"  {k}: {v}")

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
