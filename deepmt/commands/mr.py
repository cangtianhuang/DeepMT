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

from deepmt._utils import get_library, get_repo, not_implemented_error


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
        from deepmt.ir import OperatorIR
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
            click.echo(click.style(f"\n已保存 {count} 个 MR 至用户仓库", fg="cyan"))

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
        from deepmt.ir import OperatorIR
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
    click.echo(f"  总 MR 数:   {stats['total_mrs']}")
    click.echo(f"  已验证:     {stats['verified_mrs']}")
    click.echo(f"  未验证:     {stats['unverified_mrs']}")
    click.echo(f"  数值检查:   {stats['checked']}")
    click.echo(f"  符号证明:   {stats['proven']}")

    if not operator and stats["by_operator"]:
        click.echo("\n  按算子分布:")
        for op, s in stats["by_operator"].items():
            click.echo(f"    {op}: total={s['total']}, verified={s['verified']}")


# ── 辅助函数 ─────────────────────────────────────────────────────────────────


def _collect_batch_operators(framework: str, category_filter=None):
    """
    构建 batch-generate 的算子列表，返回 [(operator_name, category), ...] 。

    来源：mr_templates.yaml 的 operator_mr_mapping（键为泛化短名，如 relu/abs）。
    framework 参数用于设置生成 MR 的 applicable_frameworks，不做名称过滤。
    若指定 category_filter，按对应模板的 category 字段过滤。
    """
    from deepmt.mr_generator.base.mr_templates import MRTemplatePool

    pool = MRTemplatePool()

    result = []
    seen = set()

    for op_name in pool.operator_mr_mapping:
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

    支持两种格式：
      - 泛化短名（如 relu）：委托框架插件的 _resolve_operator 解析
      - 完整 Python 路径（如 torch.nn.functional.relu）：直接 importlib 导入
    返回可调用对象，失败则返回 None。
    """
    try:
        if "." not in operator_name:
            # 泛化短名：委托插件解析（复用 _overrides + 根模块回退逻辑）
            from deepmt.core.plugins_manager import get_plugins_manager
            backend = get_plugins_manager().get_backend(framework)
            return backend._resolve_operator(operator_name)
        else:
            # 全路径名：直接 importlib 导入
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
    "--operator", "-o",
    "operators",
    multiple=True,
    help="显式指定要处理的算子（可重复，如 -o relu -o abs）；"
         "未在 operator_mr_mapping 中注册的算子将自动进入发现模式",
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
def mr_batch_generate(framework, operators, category, limit, skip_existing, sources, precheck, sympy, dry_run):
    """批量为算子目录中的算子生成蜕变关系并保存至知识库。

    当通过 -o/--operator 显式指定算子且该算子不在 operator_mr_mapping 中时，
    自动进入"发现模式"：遍历全部模板，用 precheck 筛选适用的模板，
    并将结果写回 mr_templates.yaml 的 operator_mr_mapping，实现从零重建知识库。

    \b
    示例:
      deepmt mr batch-generate --framework pytorch --category activation
      deepmt mr batch-generate --limit 10 --skip-existing
      deepmt mr batch-generate --dry-run
      deepmt mr batch-generate --sources llm,template --precheck
      deepmt mr batch-generate -o relu -o abs -o exp -o sigmoid --sources template --precheck --no-sympy
    """
    source_list = [s.strip().lower() for s in sources.split(",")]
    invalid = set(source_list) - {"llm", "template"}
    if invalid:
        raise click.BadParameter(f"未知来源: {', '.join(invalid)}，可选 llm / template")

    from deepmt.mr_generator.base.mr_templates import MRTemplatePool
    pool = MRTemplatePool()

    # ── 构建算子列表 ──────────────────────────────────────────────────────────
    if operators:
        # 显式指定算子：不需要 operator_mr_mapping 有内容
        # 未注册的算子标记为 "discover" 分类，后续进入发现模式
        operator_list = []
        for op in operators:
            if op in pool.operator_mr_mapping:
                # 已注册：用第一个模板的 category 作为标签
                tnames = pool.operator_mr_mapping[op]
                cat = pool.templates[tnames[0]].category if tnames and tnames[0] in pool.templates else "registered"
            else:
                cat = "discover"
            operator_list.append((op, cat))
    else:
        operator_list = _collect_batch_operators(framework, category)

    if not operator_list:
        click.echo(click.style(
            f"未找到符合条件的算子（framework={framework}"
            + (f", category={category}" if category else "")
            + ("，请通过 -o/--operator 显式指定算子" if not operators else "") + "）",
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
        from deepmt.ir import OperatorIR
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

                # ── 发现模式：算子未在 operator_mr_mapping 中注册 ──────────────
                is_discover = op_category == "discover"
                discover_pairs = []   # [(MRTemplate, MetamorphicRelation), ...]

                if is_discover and "template" in source_list:
                    # 遍历全部模板，由 precheck 决定哪些适用
                    discover_templates = pool.discover_all_templates(operator_func)
                    click.echo(
                        f"{prefix}  [discover] 尝试 {len(discover_templates)} 个候选模板 …"
                    )
                    discover_pairs = [
                        (t, pool.create_mr_from_template(t))
                        for t in discover_templates
                    ]
                    # 合并：LLM 来源正常走 generate_only，template 来源用 discover_pairs
                    llm_mrs = []
                    if "llm" in source_list:
                        llm_mrs = generator.generate_only(
                            operator_ir=operator_ir,
                            auto_fetch_info=False,
                            sources=["llm"],
                            framework=framework,
                        )
                    mrs = llm_mrs + [mr for _, mr in discover_pairs]
                else:
                    # 常规模式：从 operator_mr_mapping 取模板
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

                # ── 发现模式：将通过 precheck 的模板写回 operator_mr_mapping ──
                if is_discover and discover_pairs and precheck:
                    passing_names = [
                        t.name for t, mr in discover_pairs if mr.checked is True
                    ]
                    if passing_names:
                        pool.update_operator_mapping(operator, passing_names)
                        click.echo(
                            f"{prefix}  [discover] 发现 {len(passing_names)} 个适用模板 "
                            f"→ 已写入 operator_mr_mapping"
                        )
                    else:
                        click.echo(
                            f"{prefix}  [discover] precheck 全部失败，"
                            "operator_mr_mapping 未更新"
                        )

                passed = sum(1 for m in mrs if m.verified)
                count = repo.save(operator, mrs, framework=framework)

                if count == 0 and not mrs:
                    click.echo(f"{prefix}  {click.style('NO_MR', fg='yellow')} (无候选 MR)")
                    skipped += 1
                elif passed == 0:
                    click.echo(
                        f"{prefix}  {click.style('NO_MR', fg='yellow')} "
                        f"(生成={len(mrs)} 验证通过=0)"
                    )
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


# ── promote ───────────────────────────────────────────────────────────────────

@mr.command("promote")
@click.argument("operator")
@click.option("--layer", default="operator", show_default=True, help="MR 层次")
def mr_promote(operator, layer):
    """将用户仓库中 verified=True 的 MR 迁移到项目库。

    \b
    示例:
      deepmt mr promote torch.add
    """
    repo = get_repo()
    if not repo.exists(operator):
        click.echo(click.style(f"用户仓库中未找到算子 '{operator}' 的 MR", fg="yellow"))
        sys.exit(1)

    lib = get_library(layer=layer)
    count = lib.promote_from_repository(operator, repo)
    if count:
        click.echo(click.style(f"已迁移 {count} 个 MR 到项目库", fg="green"))
    else:
        click.echo(click.style("没有 verified=True 的 MR 可迁移", fg="yellow"))


# ── model-generate ────────────────────────────────────────────────────────────

@mr.command("model-generate")
@click.argument("model_name", required=False, default=None)
@click.option(
    "--framework",
    default="pytorch",
    type=click.Choice(["pytorch"], case_sensitive=False),
    show_default=True,
    help="目标框架",
)
@click.option("--max-mrs", default=None, type=int, help="每个模型最多生成的 MR 数量")
@click.option("--all", "gen_all", is_flag=True, default=False, help="为所有基准模型生成 MR")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def mr_model_generate(model_name, framework, max_mrs, gen_all, as_json):
    """为模型层基准对象生成蜕变关系（基于结构分析，无需 LLM）。

    系统自动分析模型结构，根据模型类型（MLP/CNN/RNN/Transformer）
    选择适用的变换策略，生成对应的模型层 MR 列表。

    \b
    示例:
      deepmt mr model-generate SimpleMLP              # 为 SimpleMLP 生成 MR
      deepmt mr model-generate --all                  # 为所有基准模型生成 MR
      deepmt mr model-generate SimpleCNN --max-mrs 5
      deepmt mr model-generate SimpleMLP --json
    """
    try:
        from deepmt.benchmarks.models import ModelBenchmarkRegistry
        from deepmt.mr_generator.model import ModelMRGenerator

        registry = ModelBenchmarkRegistry()
        generator = ModelMRGenerator()

        if gen_all:
            names = registry.names(framework=framework)
        elif model_name:
            names = [model_name]
        else:
            click.echo(
                click.style(
                    "请指定模型名称，或使用 --all 为所有基准模型生成 MR。",
                    fg="yellow",
                ),
                err=True,
            )
            sys.exit(1)

        all_results = {}
        for name in names:
            model_ir = registry.get(name, with_instance=True)
            if model_ir is None:
                click.echo(click.style(f"未找到模型: {name!r}", fg="yellow"), err=True)
                continue
            mrs = generator.generate(model_ir, max_per_model=max_mrs)
            all_results[name] = mrs
            if not as_json:
                click.echo(f"\n{name} ({model_ir.model_type})  — {len(mrs)} MRs 生成")
                for mr in mrs:
                    click.echo(
                        f"  [{mr.id[:8]}] {mr.description[:65]}"
                    )
                    click.echo(
                        f"           transform: {mr.transform_code[:55]}"
                    )
                    click.echo(
                        f"           oracle:    {mr.oracle_expr}"
                    )

        if as_json:
            output = {
                name: [
                    {
                        "id": mr.id,
                        "description": mr.description,
                        "transform_code": mr.transform_code,
                        "oracle_expr": mr.oracle_expr,
                        "category": mr.category,
                        "subject_name": mr.subject_name,
                    }
                    for mr in mrs
                ]
                for name, mrs in all_results.items()
            }
            click.echo(json.dumps(output, ensure_ascii=False, indent=2))

    except Exception as e:
        click.echo(click.style(f"错误: {e}", fg="red"), err=True)
        sys.exit(1)
