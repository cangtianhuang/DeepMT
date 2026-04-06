"""
deepmt mr — MR 生成与管理子命令组

命令:
    generate    为算子生成蜕变关系（仅算子层已实现）
    verify      对知识库中的 MR 进行验证
    list        列出算子的所有 MR
    stats       显示 MR 知识库统计信息
    delete      删除算子的 MR 记录
"""

import json
import sys

import click

from deepmt._utils import get_repo, not_implemented_error


@click.group()
def mr():
    """MR 生成与管理（算子层）。"""


# ── generate ─────────────────────────────────────────────────────────────────

@mr.command("generate")
@click.argument("operator")
@click.option(
    "--layer",
    default="operator",
    type=click.Choice(["operator", "model", "application"], case_sensitive=False),
    show_default=True,
    help="测试层次",
)
@click.option(
    "--framework",
    default="pytorch",
    type=click.Choice(["pytorch", "tensorflow", "paddlepaddle"], case_sensitive=False),
    show_default=True,
    help="目标框架",
)
@click.option(
    "--sources",
    default="llm,template",
    show_default=True,
    help="MR 生成来源，逗号分隔（可选值: llm, template）",
)
@click.option("--precheck/--no-precheck", default=True, show_default=True, help="启用数值预检")
@click.option("--sympy/--no-sympy", default=False, show_default=True, help="启用 SymPy 符号证明")
@click.option("--auto-fetch/--no-auto-fetch", default=False, show_default=True, help="自动从网络获取算子文档")
@click.option("--save/--no-save", default=True, show_default=True, help="将结果保存至知识库")
@click.option("--version", "ver", default=1, show_default=True, type=int, help="知识库版本号")
def mr_generate(operator, layer, framework, sources, precheck, sympy, auto_fetch, save, ver):
    """为算子生成蜕变关系并（可选）保存至知识库。

    \b
    示例:
      deepmt mr generate relu
      deepmt mr generate relu --sources llm --precheck --sympy --save
      deepmt mr generate relu --no-auto-fetch --no-sympy
    """
    if layer != "operator":
        not_implemented_error(f"--layer {layer}", "仅算子层（operator）已实现，模型层与应用层尚在开发中。")

    source_list = [s.strip().lower() for s in sources.split(",")]
    invalid = set(source_list) - {"llm", "template"}
    if invalid:
        raise click.BadParameter(f"未知来源: {', '.join(invalid)}，可选 llm / template")

    click.echo(f"[generate] 算子: {operator}  框架: {framework}  来源: {source_list}")
    click.echo(f"           precheck={precheck}  sympy={sympy}  auto_fetch={auto_fetch}  save={save}")

    try:
        from deepmt.ir.schema import OperatorIR
        from deepmt.mr_generator.operator.operator_mr import OperatorMRGenerator
        from deepmt.mr_generator.base.mr_repository import MRRepository

        operator_ir = OperatorIR(name=operator, inputs=[])
        generator = OperatorMRGenerator()

        click.echo("正在生成候选 MR …")
        mrs = generator.generate_only(
            operator_ir=operator_ir,
            auto_fetch_info=auto_fetch,
            sources=source_list,
            framework=framework,
        )
        click.echo(f"生成候选: {len(mrs)} 个")

        if precheck or sympy:
            click.echo("正在验证 MR …")
            mrs = generator.verify_mrs(
                mrs=mrs,
                operator_ir=operator_ir,
                operator_func=None,
                use_precheck=precheck,
                use_sympy_proof=sympy,
            )

        passed = sum(1 for m in mrs if m.verified)
        click.echo(f"验证通过: {passed}/{len(mrs)}")

        for m in mrs:
            mark = click.style("✓", fg="green") if m.verified else click.style("✗", fg="red")
            click.echo(f"  [{mark}] {m.description}")

        if save and mrs:
            repo = get_repo()
            count = repo.save(operator, mrs, version=ver, framework=framework)
            click.echo(click.style(f"\n已保存 {count} 个 MR 至知识库（version={ver}）", fg="cyan"))

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)


# ── verify ────────────────────────────────────────────────────────────────────

@mr.command("verify")
@click.argument("operator")
@click.option("--version", "ver", default=None, type=int, help="版本号（默认: 最新）")
@click.option("--precheck/--no-precheck", default=True, show_default=True, help="启用数值预检")
@click.option("--sympy/--no-sympy", default=False, show_default=True, help="启用 SymPy 符号证明")
@click.option("--save/--no-save", default=False, show_default=True, help="将验证结果更新到知识库")
def mr_verify(operator, ver, precheck, sympy, save):
    """对知识库中已有的 MR 执行验证。

    \b
    示例:
      deepmt mr verify relu
      deepmt mr verify relu --sympy --version 1
    """
    repo = get_repo()
    if not repo.exists(operator, ver):
        click.echo(click.style(f"知识库中未找到算子 '{operator}' 的 MR（version={ver}）", fg="yellow"))
        sys.exit(1)

    click.echo(f"[verify] 算子: {operator}  version={ver}  precheck={precheck}  sympy={sympy}")

    try:
        from deepmt.ir.schema import OperatorIR
        from deepmt.mr_generator.operator.operator_mr import OperatorMRGenerator

        operator_ir = OperatorIR(name=operator, inputs=[])
        generator = OperatorMRGenerator()

        mrs = repo.load(operator, ver)
        click.echo(f"从知识库加载: {len(mrs)} 个 MR")

        mrs = generator.verify_mrs(
            mrs=mrs,
            operator_ir=operator_ir,
            operator_func=None,
            use_precheck=precheck,
            use_sympy_proof=sympy,
        )

        passed = sum(1 for m in mrs if m.verified)
        click.echo(f"验证通过: {passed}/{len(mrs)}")
        for m in mrs:
            mark = click.style("✓", fg="green") if m.verified else click.style("✗", fg="red")
            click.echo(f"  [{mark}] {m.description}")

        if save:
            repo.save(operator, mrs, version=ver if ver else 1)
            click.echo(click.style("验证结果已更新至知识库", fg="cyan"))

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)


# ── list ──────────────────────────────────────────────────────────────────────

@mr.command("list")
@click.argument("operator", required=False, default=None)
@click.option("--version", "ver", default=None, type=int, help="版本号（默认: 最新）")
@click.option("--verified-only", is_flag=True, default=False, help="仅显示已验证的 MR")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def mr_list(operator, ver, verified_only, as_json):
    """列出算子的蜕变关系。

    OPERATOR 可省略，省略时列出所有算子名称。

    \b
    示例:
      deepmt mr list
      deepmt mr list relu
      deepmt mr list relu --version 1 --verified-only
      deepmt mr list relu --json
    """
    repo = get_repo()

    if operator is None:
        ops = repo.list_operators()
        if not ops:
            click.echo("知识库为空，尚未保存任何 MR。")
            return
        click.echo(f"知识库中共有 {len(ops)} 个算子:")
        for op in ops:
            versions = repo.get_versions(op)
            stats = repo.get_statistics(op)
            click.echo(f"  {op}  (versions: {versions}, total: {stats['total_mrs']}, verified: {stats['verified_mrs']})")
        return

    if not repo.exists(operator, ver):
        click.echo(click.style(f"知识库中未找到算子 '{operator}' 的 MR", fg="yellow"))
        sys.exit(1)

    mrs = repo.get_mr_with_validation_status(operator, ver, verified_only=verified_only)

    if as_json:
        data = [
            {
                "id": m.id,
                "description": m.description,
                "category": m.category,
                "oracle_expr": m.oracle_expr,
                "transform_code": m.transform_code,
                "verified": m.verified,
            }
            for m in mrs
        ]
        click.echo(json.dumps(data, ensure_ascii=False, indent=2))
        return

    label = f"算子 '{operator}'" + (f" version={ver}" if ver else " (最新版本)")
    click.echo(f"{label} — {len(mrs)} 个 MR:")
    for m in mrs:
        mark = click.style("✓", fg="green") if m.verified else click.style("✗", fg="red")
        click.echo(f"  [{mark}] [{m.category}] {m.description}")
        click.echo(f"       oracle: {m.oracle_expr}")
        if m.transform_code:
            click.echo(f"       transform: {m.transform_code}")


# ── stats ─────────────────────────────────────────────────────────────────────

@mr.command("stats")
@click.argument("operator", required=False, default=None)
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def mr_stats(operator, as_json):
    """显示 MR 知识库统计信息。

    \b
    示例:
      deepmt mr stats
      deepmt mr stats relu
      deepmt mr stats --json
    """
    repo = get_repo()
    stats = repo.get_statistics(operator)

    if as_json:
        click.echo(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    title = f"算子 '{operator}' 的统计" if operator else "知识库整体统计"
    click.echo(f"\n{title}")
    click.echo("─" * 40)
    click.echo(f"  总 MR 数:      {stats['total_mrs']}")
    click.echo(f"  已验证:        {stats['verified_mrs']}")
    click.echo(f"  未验证:        {stats['unverified_mrs']}")
    click.echo(f"  Precheck 通过: {stats['precheck_passed']}")
    click.echo(f"  SymPy 证明:    {stats['sympy_proven']}")

    if not operator and stats["by_operator"]:
        click.echo("\n  按算子分布:")
        for op, s in stats["by_operator"].items():
            click.echo(f"    {op}: total={s['total']}, verified={s['verified']}")


# ── delete ────────────────────────────────────────────────────────────────────

@mr.command("delete")
@click.argument("operator")
@click.option("--version", "ver", default=None, type=int, help="版本号（默认: 所有版本）")
@click.option("--yes", "-y", is_flag=True, default=False, help="跳过确认提示")
def mr_delete(operator, ver, yes):
    """删除知识库中算子的 MR 记录。

    \b
    示例:
      deepmt mr delete relu
      deepmt mr delete relu --version 1 -y
    """
    not_implemented_error(
        "deepmt mr delete",
        "MR 删除功能尚未实现。如需清理，可直接删除或操作 data/mr_knowledge_base.db 文件。",
    )
