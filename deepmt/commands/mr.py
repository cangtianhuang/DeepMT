"""
deepmt mr — MR 生成与管理子命令组

命令:
    generate        为算子生成蜕变关系（仅算子层已实现）
    batch-generate  批量为目录中的算子生成蜕变关系
    verify          对知识库中的 MR 进行验证
    list            列出算子的所有 MR
    stats           显示 MR 知识库统计信息
    delete          删除算子的 MR 记录
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
def mr_generate(operator, layer, framework, sources, precheck, sympy, auto_fetch, save):
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
        from deepmt.mr_generator.operator.operator_mr_generator import OperatorMRGenerator

        operator_func = _try_import_operator(operator, framework)
        if operator_func:
            click.echo(f"           operator_func=已加载 ({operator})")
        else:
            click.echo(f"           operator_func=未找到（precheck 将跳过）")

        operator_ir = OperatorIR(name=operator)
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
                framework=framework,
                operator_func=operator_func,
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
            count = repo.save(operator, mrs, framework=framework)
            click.echo(click.style(f"\n已保存 {count} 个 MR 至知识库", fg="cyan"))

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)


# ── verify ────────────────────────────────────────────────────────────────────

@mr.command("verify")
@click.argument("operator")
@click.option(
    "--framework",
    default="pytorch",
    type=click.Choice(["pytorch", "tensorflow", "paddlepaddle"], case_sensitive=False),
    show_default=True,
    help="目标框架",
)
@click.option("--precheck/--no-precheck", default=True, show_default=True, help="启用数值预检")
@click.option("--sympy/--no-sympy", default=False, show_default=True, help="启用 SymPy 符号证明")
@click.option("--save/--no-save", default=False, show_default=True, help="将验证结果更新到知识库")
def mr_verify(operator, framework, precheck, sympy, save):
    """对知识库中已有的 MR 执行验证。

    \b
    示例:
      deepmt mr verify relu
      deepmt mr verify relu --sympy
    """
    repo = get_repo()
    if not repo.exists(operator):
        click.echo(click.style(f"知识库中未找到算子 '{operator}' 的 MR", fg="yellow"))
        sys.exit(1)

    click.echo(f"[verify] 算子: {operator}  precheck={precheck}  sympy={sympy}")

    try:
        from deepmt.ir.schema import OperatorIR
        from deepmt.mr_generator.operator.operator_mr_generator import OperatorMRGenerator

        operator_ir = OperatorIR(name=operator)
        generator = OperatorMRGenerator()

        mrs = repo.load(operator)
        click.echo(f"从知识库加载: {len(mrs)} 个 MR")

        mrs = generator.verify_mrs(
            mrs=mrs,
            operator_ir=operator_ir,
            framework=framework,
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
            repo.save(operator, mrs)
            click.echo(click.style("验证结果已更新至知识库", fg="cyan"))

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)


# ── list ──────────────────────────────────────────────────────────────────────

@mr.command("list")
@click.argument("operator", required=False, default=None)
@click.option("--verified-only", is_flag=True, default=False, help="仅显示已验证的 MR")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def mr_list(operator, verified_only, as_json):
    """列出算子的蜕变关系。

    OPERATOR 可省略，省略时列出所有算子名称。

    \b
    示例:
      deepmt mr list
      deepmt mr list relu
      deepmt mr list relu --verified-only
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
            stats = repo.get_statistics(op)
            click.echo(f"  {op}  (total: {stats['total_mrs']}, verified: {stats['verified_mrs']})")
        return

    if not repo.exists(operator):
        click.echo(click.style(f"知识库中未找到算子 '{operator}' 的 MR", fg="yellow"))
        sys.exit(1)

    mrs = repo.get_mr_with_validation_status(operator, verified_only=verified_only)

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

    click.echo(f"算子 '{operator}' — {len(mrs)} 个 MR:")
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


# ── 辅助函数 ─────────────────────────────────────────────────────────────────


def _collect_batch_operators(framework: str, category_filter=None):
    """
    构建 batch-generate 的算子列表，返回 [(operator_name, category), ...] 。

    来源优先级：
    1. mr_templates.yaml 的 operator_mr_mapping（有已知模板的算子）
       - 按 framework 前缀过滤（pytorch -> "torch." 开头）
       - 若指定 category_filter，按对应模板的 category 字段过滤
    2. 若无结果，回退到算子目录的 name 字段
    """
    from deepmt.mr_generator.base.mr_templates import MRTemplatePool

    pool = MRTemplatePool()

    # framework -> operator 名称前缀映射
    _FRAMEWORK_PREFIX = {
        "pytorch": ("torch.",),
        "tensorflow": ("tf.", "tensorflow."),
        "paddlepaddle": ("paddle.",),
    }
    prefixes = _FRAMEWORK_PREFIX.get(framework, ())

    result = []
    seen = set()

    for op_name in pool.operator_mr_mapping:
        # framework 过滤：通过前缀匹配
        if prefixes and not any(op_name.startswith(p) for p in prefixes):
            continue

        # category 过滤：用该算子第一个模板的 category 判断
        op_category = ""
        if category_filter:
            template_names = pool.operator_mr_mapping[op_name]
            matched = False
            for tname in template_names:
                t = pool.templates.get(tname)
                if t and t.category == category_filter:
                    op_category = t.category
                    matched = True
                    break
            if not matched:
                continue
        else:
            # 取第一个模板的 category 作为标记
            template_names = pool.operator_mr_mapping[op_name]
            if template_names:
                t = pool.templates.get(template_names[0])
                if t:
                    op_category = t.category

        if op_name not in seen:
            result.append((op_name, op_category))
            seen.add(op_name)

    return result


def _try_import_operator(operator_name: str, framework: str):
    """
    尝试按算子名称动态导入算子函数。

    支持完整 Python 路径（如 torch.nn.functional.relu）。
    返回可调用对象，失败则返回 None。
    """
    try:
        parts = operator_name.rsplit(".", 1)
        if len(parts) == 2:
            module_path, func_name = parts
            import importlib
            module = importlib.import_module(module_path)
            func = getattr(module, func_name, None)
            if callable(func):
                return func
    except Exception:
        pass
    return None


# ── batch-generate ───────────────────────────────────────────────────────────

@mr.command("batch-generate")
@click.option(
    "--framework",
    default="pytorch",
    type=click.Choice(["pytorch", "tensorflow", "paddlepaddle"], case_sensitive=False),
    show_default=True,
    help="目标框架",
)
@click.option(
    "--category",
    default=None,
    help="只处理指定分类的算子（如 activation、normalization），不指定则处理所有算子",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="最多处理 N 个算子（不指定则处理全部）",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="跳过知识库中已有 MR 的算子（支持断点续跑）",
)
@click.option(
    "--sources",
    default="template",
    show_default=True,
    help="MR 生成来源，逗号分隔（可选值: llm, template）",
)
@click.option("--precheck/--no-precheck", default=True, show_default=True, help="启用数值预检")
@click.option("--sympy/--no-sympy", default=False, show_default=True, help="启用 SymPy 符号证明")
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="只列出待处理算子，不实际执行生成",
)
def mr_batch_generate(framework, category, limit, skip_existing, sources, precheck, sympy, dry_run):
    """批量为算子目录中的算子生成蜕变关系并保存至知识库。

    \b
    示例:
      deepmt mr batch-generate --framework pytorch --category activation
      deepmt mr batch-generate --limit 10 --skip-existing
      deepmt mr batch-generate --dry-run
      deepmt mr batch-generate --sources llm,template --precheck
    """
    source_list = [s.strip().lower() for s in sources.split(",")]
    invalid = set(source_list) - {"llm", "template"}
    if invalid:
        raise click.BadParameter(f"未知来源: {', '.join(invalid)}，可选 llm / template")

    operator_list = _collect_batch_operators(framework, category)

    if not operator_list:
        click.echo(click.style(
            f"未找到符合条件的算子（framework={framework}"
            + (f", category={category}" if category else "") + "）",
            fg="yellow"
        ))
        return

    if limit is not None and limit > 0:
        operator_list = operator_list[:limit]

    click.echo(
        f"[batch-generate] framework={framework}  category={category or '全部'}"
        f"  sources={source_list}  precheck={precheck}  sympy={sympy}"
        f"  skip_existing={skip_existing}  dry_run={dry_run}"
    )
    click.echo(f"待处理算子: {len(operator_list)} 个")

    if dry_run:
        click.echo("\n[dry-run] 将处理以下算子:")
        for op_name, op_category in operator_list:
            click.echo(f"  {op_name}  [{op_category}]")
        return

    try:
        from deepmt.ir.schema import OperatorIR
        from deepmt.mr_generator.operator.operator_mr_generator import OperatorMRGenerator
        repo = get_repo()
        generator = OperatorMRGenerator()
    except Exception as e:
        click.echo(click.style(f"初始化失败: {e}", fg="red"), err=True)
        sys.exit(1)

    total = len(operator_list)
    done = 0
    skipped = 0
    failed = 0

    click.echo("")

    try:
        for idx, (operator, op_category) in enumerate(operator_list, 1):
            prefix = f"[{idx:>3}/{total}] {operator}"

            if skip_existing and repo.exists(operator):
                click.echo(f"{prefix}  {click.style('SKIP', fg='yellow')}")
                skipped += 1
                continue

            try:
                operator_func = _try_import_operator(operator, framework)
                operator_ir = OperatorIR(name=operator)

                mrs = generator.generate_only(
                    operator_ir=operator_ir,
                    auto_fetch_info=False,
                    sources=source_list,
                    framework=framework,
                )

                if not mrs:
                    click.echo(f"{prefix}  {click.style('NO_MR', fg='yellow')} (无候选 MR)")
                    skipped += 1
                    continue

                if precheck or sympy:
                    mrs = generator.verify_mrs(
                        mrs=mrs,
                        operator_ir=operator_ir,
                        framework=framework,
                        operator_func=operator_func,
                        use_precheck=precheck,
                        use_sympy_proof=sympy,
                    )

                passed = sum(1 for m in mrs if m.verified)
                count = repo.save(operator, mrs, framework=framework)

                if passed == 0:
                    click.echo(f"{prefix}  {click.style('NO_MR', fg='yellow')} (生成={len(mrs)} 验证通过=0)")
                    skipped += 1
                else:
                    click.echo(
                        f"{prefix}  {click.style('OK', fg='green')}"
                        f"  生成={len(mrs)} 验证={passed} 保存={count}"
                    )
                    done += 1

            except KeyboardInterrupt:
                raise
            except Exception as e:
                click.echo(f"{prefix}  {click.style('FAIL', fg='red')}  {e}")
                failed += 1

    except KeyboardInterrupt:
        click.echo(click.style("\n用户中断（Ctrl+C）", fg="yellow"))

    click.echo("\n" + "─" * 50)
    click.echo(
        f"完成: {click.style(str(done), fg='green')}  "
        f"跳过: {click.style(str(skipped), fg='yellow')}  "
        f"失败: {click.style(str(failed), fg='red')}  "
        f"共: {total}"
    )


# ── delete ────────────────────────────────────────────────────────────────────

@mr.command("delete")
@click.argument("operator")
@click.option("--yes", "-y", is_flag=True, default=False, help="跳过确认提示")
def mr_delete(operator, yes):
    """删除知识库中算子的 MR 记录。

    \b
    示例:
      deepmt mr delete relu
      deepmt mr delete relu -y
    """
    not_implemented_error(
        "deepmt mr delete",
        "MR 删除功能尚未实现。如需清理，可直接删除 data/mr_repository/<operator>.yaml 文件。",
    )
