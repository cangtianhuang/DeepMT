"""
deepmt catalog — 算子目录管理子命令组

命令:
    list             列出指定框架的算子（支持按分类筛选）
    search           跨框架模糊搜索算子
    info             查询某算子的跨框架分布及知识库 MR 数量
    latest-version   快速获取框架最新稳定版本号（无需 LLM）
    fetch-doc        获取并打印算子文档正文
    import-from-docs 从官方文档批量导入 API 到算子目录
    update-api-list  从官方文档页面更新 API 模块列表缓存
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import click
import yaml

_CATALOG_DIR = Path(__file__).parent.parent.parent / "mr_generator" / "config" / "operator_catalog"
_ALL_FRAMEWORKS = ["pytorch", "tensorflow", "paddlepaddle"]


def _load_catalog(framework: str) -> Optional[Dict]:
    """加载指定框架的算子目录 YAML，若文件不存在返回 None。"""
    path = _CATALOG_DIR / f"{framework}.yaml"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_operators(framework: str) -> List[Dict]:
    """返回指定框架的算子列表（字典列表）。"""
    catalog = _load_catalog(framework)
    if catalog is None:
        return []
    return catalog.get("operators", [])


def _match(op: Dict, keyword: str) -> bool:
    """判断算子是否匹配关键字（名称或别名包含 keyword，大小写不敏感）。"""
    kw = keyword.lower()
    if kw in op.get("name", "").lower():
        return True
    for alias in op.get("aliases", []):
        if kw in alias.lower():
            return True
    return False


@click.group()
def catalog():
    """算子目录管理（列出/搜索框架算子，查询 MR 覆盖情况）。"""


# ── list ──────────────────────────────────────────────────────────────────────

@catalog.command("list")
@click.option(
    "--framework", "-f",
    default="pytorch",
    type=click.Choice(_ALL_FRAMEWORKS, case_sensitive=False),
    show_default=True,
    help="目标框架",
)
@click.option(
    "--category", "-c",
    default=None,
    metavar="CATEGORY",
    help="按分类筛选（如 activation / math / pooling …）",
)
@click.option(
    "--search", "-s",
    default=None,
    metavar="KEYWORD",
    help="按名称关键字过滤（模糊匹配，含别名）",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
@click.option("--count-only", is_flag=True, default=False, help="仅输出匹配算子数量")
def catalog_list(framework, category, search, as_json, count_only):
    """列出指定框架的算子，支持按分类/关键字筛选。

    \b
    示例:
      deepmt catalog list --framework pytorch
      deepmt catalog list --framework pytorch --category activation
      deepmt catalog list --framework pytorch --search relu
      deepmt catalog list --framework tensorflow --search conv --json
      deepmt catalog list --framework pytorch --count-only
    """
    ops = _get_operators(framework)
    if not ops:
        click.echo(click.style(f"框架 '{framework}' 的算子目录为空或尚未配置。", fg="yellow"))
        sys.exit(1)

    # 筛选
    if category:
        ops = [o for o in ops if o.get("category", "").lower() == category.lower()]
    if search:
        ops = [o for o in ops if _match(o, search)]

    if count_only:
        click.echo(str(len(ops)))
        return

    if as_json:
        click.echo(json.dumps(ops, ensure_ascii=False, indent=2))
        return

    filter_desc = ""
    if category:
        filter_desc += f"  分类={category}"
    if search:
        filter_desc += f"  搜索='{search}'"
    click.echo(f"\n框架: {framework}{filter_desc}  共 {len(ops)} 个算子\n")

    # 按 category 分组显示
    categories: Dict[str, List[Dict]] = {}
    for op in ops:
        cat = op.get("category", "other")
        categories.setdefault(cat, []).append(op)

    for cat, cat_ops in sorted(categories.items()):
        click.echo(f"  [{cat}]")
        for op in cat_ops:
            aliases = op.get("aliases", [])
            alias_str = f"  → {', '.join(aliases)}" if aliases else ""
            note = f"  # {op['note']}" if op.get("note") else ""
            deprecated = click.style(" [deprecated]", fg="yellow") if op.get("deprecated") else ""
            click.echo(f"    {op['name']}{deprecated}{alias_str}{note}")


# ── search ────────────────────────────────────────────────────────────────────

@catalog.command("search")
@click.argument("keyword")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def catalog_search(keyword, as_json):
    """跨所有框架搜索算子（模糊匹配名称及别名）。

    \b
    示例:
      deepmt catalog search relu
      deepmt catalog search conv
      deepmt catalog search batch_norm --json
    """
    results: Dict[str, List[Dict]] = {}
    for fw in _ALL_FRAMEWORKS:
        matched = [o for o in _get_operators(fw) if _match(o, keyword)]
        if matched:
            results[fw] = matched

    if not results:
        click.echo(f"未找到与 '{keyword}' 匹配的算子。")
        return

    if as_json:
        click.echo(json.dumps(results, ensure_ascii=False, indent=2))
        return

    total = sum(len(v) for v in results.values())
    click.echo(f"\n搜索 '{keyword}'：在 {len(results)} 个框架中找到 {total} 个匹配算子\n")
    for fw, ops in results.items():
        click.echo(f"  [{fw}]  {len(ops)} 个:")
        for op in ops:
            aliases = op.get("aliases", [])
            alias_str = f"  → {', '.join(aliases)}" if aliases else ""
            click.echo(f"    {op['name']}  ({op.get('category', '?')}){alias_str}")


# ── info ──────────────────────────────────────────────────────────────────────

@catalog.command("info")
@click.argument("operator")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def catalog_info(operator, as_json):
    """查询算子的跨框架分布及知识库 MR 数量。

    显示内容:
      - 算子是否存在于各框架的算子目录（精确名称 + 别名匹配）
      - 知识库（SQLite）中该算子存储的 MR 数量及验证情况

    \b
    示例:
      deepmt catalog info relu
      deepmt catalog info torch.nn.ReLU
      deepmt catalog info relu --json
    """
    # 1. 跨框架目录查询
    catalog_presence: Dict[str, Optional[Dict]] = {}
    for fw in _ALL_FRAMEWORKS:
        ops = _get_operators(fw)
        found = next(
            (o for o in ops if _match(o, operator) or o.get("name", "").lower() == operator.lower()),
            None,
        )
        catalog_presence[fw] = found

    # 2. 知识库 MR 查询
    try:
        from deepmt.mr_generator.base.mr_repository import MRRepository
        repo = MRRepository()
        has_mr = repo.exists(operator)
        mr_stats = repo.get_statistics(operator) if has_mr else None
        versions = repo.get_versions(operator) if has_mr else []
    except Exception as e:
        has_mr = False
        mr_stats = None
        versions = []
        click.echo(click.style(f"[warn] 知识库查询失败: {e}", fg="yellow"), err=True)

    if as_json:
        result = {
            "operator": operator,
            "catalog": {
                fw: {
                    "found": entry is not None,
                    "name": entry.get("name") if entry else None,
                    "category": entry.get("category") if entry else None,
                    "since": entry.get("since") if entry else None,
                    "aliases": entry.get("aliases", []) if entry else [],
                }
                for fw, entry in catalog_presence.items()
            },
            "knowledge_base": {
                "has_mr": has_mr,
                "versions": versions,
                **(mr_stats if mr_stats else {}),
            },
        }
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return

    click.echo(f"\n算子: {click.style(operator, bold=True)}")

    # 目录分布
    click.echo("\n  框架目录分布:")
    any_found = False
    for fw, entry in catalog_presence.items():
        if entry:
            any_found = True
            cat = entry.get("category", "?")
            since = entry.get("since", "?")
            aliases = entry.get("aliases", [])
            alias_str = f"  → {', '.join(aliases)}" if aliases else ""
            click.echo(
                f"    {click.style('✓', fg='green')} {fw:<16} {entry['name']}  "
                f"[{cat}]  since={since}{alias_str}"
            )
        else:
            click.echo(f"    {click.style('✗', fg='red')} {fw:<16} 未找到")
    if not any_found:
        click.echo(click.style("    所有框架目录中均未找到该算子。请检查名称是否正确。", fg="yellow"))

    # 知识库 MR 情况
    click.echo("\n  知识库 MR 情况:")
    if has_mr and mr_stats:
        click.echo(f"    {click.style('✓', fg='green')} 已有 MR   "
                   f"versions={versions}  "
                   f"total={mr_stats['total_mrs']}  "
                   f"verified={mr_stats['verified_mrs']}")
    else:
        click.echo(f"    {click.style('—', fg='yellow')} 知识库中暂无该算子的 MR")
        click.echo(f"    提示: 运行 'deepmt mr generate {operator} --save' 生成")


# ── latest-version ─────────────────────────────────────────────────────────────

@catalog.command("latest-version")
@click.option(
    "--framework", "-f",
    default="pytorch",
    type=click.Choice(_ALL_FRAMEWORKS, case_sensitive=False),
    show_default=True,
    help="目标框架",
)
@click.option("--all-versions", is_flag=True, default=False, help="列出所有历史版本（按发布时间降序，最多显示 20 条）")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def catalog_latest_version(framework, all_versions, as_json):
    """从 PyPI 快速获取框架最新稳定版本号（无需 LLM，纯 HTTP）。

    \b
    示例:
      deepmt catalog latest-version
      deepmt catalog latest-version --framework tensorflow
      deepmt catalog latest-version --all-versions
      deepmt catalog latest-version --json
    """
    try:
        from deepmt.tools.web_search.search_agent import SearchAgent
        agent = SearchAgent()
    except Exception as e:
        click.echo(click.style(f"错误：{e}", fg="red"), err=True)
        sys.exit(1)

    if all_versions:
        click.echo(f"正在获取 {framework} 历史版本列表…", err=True)
        versions = agent.fetch_framework_versions(framework)
        if not versions:
            click.echo(click.style("获取失败，请检查网络连接。", fg="red"))
            sys.exit(1)
        if as_json:
            click.echo(json.dumps(versions[:20], ensure_ascii=False, indent=2))
        else:
            click.echo(f"\n{framework} 版本列表（最新 {min(len(versions), 20)} 条）:\n")
            for v in versions[:20]:
                click.echo(f"  {v['version']:<16}  {v['upload_time'][:10]}")
        return

    version = agent.get_latest_stable_version(framework)
    if not version:
        click.echo(click.style("获取失败，请检查网络连接。", fg="red"))
        sys.exit(1)

    if as_json:
        click.echo(json.dumps({"framework": framework, "version": version}, ensure_ascii=False))
    else:
        click.echo(f"{framework} 最新稳定版本: {click.style(version, bold=True, fg='green')}")


# ── fetch-doc ──────────────────────────────────────────────────────────────────

@catalog.command("fetch-doc")
@click.argument("operator")
@click.option(
    "--framework", "-f",
    default="pytorch",
    type=click.Choice(_ALL_FRAMEWORKS, case_sensitive=False),
    show_default=True,
    help="目标框架",
)
@click.option(
    "--url", "-u",
    default=None,
    metavar="URL",
    help="直接指定文档页面 URL（会跳过搜索步骤）",
)
@click.option("--max-chars", default=3000, show_default=True, help="最多显示字符数（0 = 不限）")
def catalog_fetch_doc(operator, framework, url, max_chars):
    """从官方文档网页获取并打印算子文档正文（无需 LLM）。

    \b
    示例:
      deepmt catalog fetch-doc torch.matmul
      deepmt catalog fetch-doc torch.matmul --url https://docs.pytorch.org/docs/stable/generated/torch.matmul.html
      deepmt catalog fetch-doc relu --framework pytorch --max-chars 0
    """
    try:
        from deepmt.tools.web_search.search_agent import SearchAgent
        agent = SearchAgent()
    except Exception as e:
        click.echo(click.style(f"错误：{e}", fg="red"), err=True)
        sys.exit(1)

    # 若未指定 URL，使用框架文档的标准 URL 格式推导
    if url is None:
        if framework == "pytorch":
            # torch.matmul -> generated/torch.matmul.html
            url = f"https://docs.pytorch.org/docs/stable/generated/{operator}.html"
        else:
            click.echo(click.style(
                f"框架 '{framework}' 需要通过 --url 指定文档链接。", fg="yellow"
            ))
            sys.exit(1)

    click.echo(f"正在获取文档: {url}", err=True)
    doc = agent.fetch_operator_doc_by_url(url)

    if not doc:
        click.echo(click.style("获取失败，请检查 URL 或网络连接。", fg="red"))
        sys.exit(1)

    if max_chars and len(doc) > max_chars:
        click.echo(doc[:max_chars])
        click.echo(click.style(f"\n… （截断，共 {len(doc)} 字符，使用 --max-chars 0 查看全部）", fg="yellow"))
    else:
        click.echo(doc)


# ── update-api-list ────────────────────────────────────────────────────────────

@catalog.command("update-api-list")
@click.option(
    "--framework", "-f",
    default="pytorch",
    type=click.Choice(["pytorch"], case_sensitive=False),
    show_default=True,
    help="目标框架（当前支持 pytorch）",
)
@click.option(
    "--version", "-v",
    default="stable",
    show_default=True,
    metavar="VERSION",
    help="文档版本，'stable' 或具体版本如 '2.1'（非 stable 暂未实现）",
)
@click.option("--no-cache", is_flag=True, default=False, help="忽略本地缓存，强制重新拉取")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出 API 列表")
@click.option("--show-cache-path", is_flag=True, default=False, help="显示缓存文件路径")
def catalog_update_api_list(framework, version, no_cache, as_json, show_cache_path):
    """从官方文档获取 API 模块列表并缓存到本地（无需 LLM）。

    从 pytorch-api.html 等页面解析所有 API 模块条目（名称 + 链接），
    保存至 tools/web_search/.cache/ 目录，后续查询直接命中缓存。

    注意：此命令获取的是模块级列表（如 torch.nn），非个体 API。
    获取个体 API 并与目录对比，请使用 check-updates 命令。

    \b
    示例:
      deepmt catalog update-api-list
      deepmt catalog update-api-list --no-cache
      deepmt catalog update-api-list --json
      deepmt catalog update-api-list --show-cache-path
    """
    try:
        from deepmt.tools.web_search.search_agent import SearchAgent, _FRAMEWORK_API_PAGES
        agent = SearchAgent()
    except Exception as e:
        click.echo(click.style(f"错误：{e}", fg="red"), err=True)
        sys.exit(1)

    try:
        _ = agent.fetch_api_list(framework=framework, version=version, use_cache=True)
    except NotImplementedError as e:
        click.echo(click.style(f"未实现：{e}", fg="yellow"))
        sys.exit(1)

    api_url = _FRAMEWORK_API_PAGES.get(framework)
    if not api_url:
        click.echo(click.style(f"框架 '{framework}' 暂不支持 API 列表更新。", fg="yellow"))
        sys.exit(1)

    use_cache = not no_cache

    if show_cache_path:
        cache_path = agent._api_list_cache_path(api_url)
        click.echo(str(cache_path))
        exists = cache_path.exists()
        status = click.style("已缓存", fg="green") if exists else click.style("未缓存", fg="yellow")
        click.echo(f"状态: {status}")
        return

    action = "使用缓存或" if use_cache else "强制"
    click.echo(f"正在{action}拉取 {framework} ({version}) API 模块列表...", err=True)

    entries = agent.fetch_api_list(framework=framework, version=version, use_cache=use_cache)

    if not entries:
        click.echo(click.style("获取失败，请检查网络连接。", fg="red"))
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(entries, ensure_ascii=False, indent=2))
        return

    click.echo(f"\n{framework} API 列表  共 {len(entries)} 条\n")
    for entry in entries:
        click.echo(f"  {entry['name']:<45}  {entry['url']}")

    # 显示缓存路径
    cache_path = agent._api_list_cache_path(api_url)
    click.echo(click.style(f"\n已缓存至: {cache_path}", fg="green"))


# ── check-updates ──────────────────────────────────────────────────────────────

@catalog.command("check-updates")
@click.option(
    "--framework", "-f",
    default="pytorch",
    type=click.Choice(["pytorch"], case_sensitive=False),
    show_default=True,
    help="目标框架（当前支持 pytorch）",
)
@click.option(
    "--version", "-v",
    default="stable",
    show_default=True,
    metavar="VERSION",
    help="文档版本，'stable' 或具体版本如 '2.1'（非 stable 暂未实现）",
)
@click.option("--no-cache", is_flag=True, default=False, help="忽略缓存，强制重新拉取文档")
@click.option("--show-no-sig", is_flag=True, default=False,
              help="显示目录中尚未记录签名的算子（默认隐藏，数量通常较多）")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出完整 diff 结果")
@click.option(
    "--skip-namespaces",
    default=None,
    metavar="NS1,NS2,...",
    help="跳过拉取这些命名空间的文档（逗号分隔，如 'torch.distributed,torch.cuda'）",
)
def catalog_check_updates(framework, version, no_cache, show_no_sig, as_json, skip_namespaces):
    """对比官方文档 API 与本地目录，报告新增/变更的 API。

    从官方文档逐页提取所有个体 API（类/函数），与本地算子目录（pytorch.yaml）
    和排除列表（pytorch_exclude.yaml）对比，输出三类差异：

    \b
    1. 目录中的算子，在文档中签名发生变化（recorded_sig != current_sig）
    2. 文档中存在，但既不在目录也不在排除列表的新 API（需要手动决定是否加入目录）
    3. 目录中的算子，在文档中未找到（可能已改名或删除）

    首次运行会拉取所有相关模块页面（约 10-30 个页面，约 1-2 分钟），
    之后 24 小时内命中缓存（毫秒级）。

    \b
    示例:
      deepmt catalog check-updates
      deepmt catalog check-updates --no-cache
      deepmt catalog check-updates --show-no-sig
      deepmt catalog check-updates --skip-namespaces torch.distributions,torch.sparse
      deepmt catalog check-updates --json > diff.json
    """
    try:
        from deepmt.tools.web_search.api_list_fetcher import APIListFetcher
        fetcher = APIListFetcher()
    except Exception as e:
        click.echo(click.style(f"错误：{e}", fg="red"), err=True)
        sys.exit(1)

    # 解析跳过的命名空间
    skip_ns_list: List = []
    if skip_namespaces:
        skip_ns_list = [ns.strip() for ns in skip_namespaces.split(",") if ns.strip()]

    # 同时加载排除列表中的命名空间，作为拉取时的跳过提示
    try:
        from deepmt.mr_generator.base.operator_catalog import OperatorCatalog
        cat = OperatorCatalog()
        exclude_cfg = cat.load_exclude_config(framework)
        # 合并用户指定的跳过和排除列表中的命名空间（用于加速拉取）
        all_skip = list(set(skip_ns_list + exclude_cfg.get("excluded_namespaces", [])))
    except Exception as e:
        click.echo(click.style(f"[warn] 无法加载目录/排除列表：{e}", fg="yellow"), err=True)
        cat = None
        all_skip = skip_ns_list

    try:
        version_guard = fetcher._check_supported  # noqa: type check
    except Exception:
        pass
    try:
        fetcher._check_supported(framework, version)
    except NotImplementedError as e:
        click.echo(click.style(f"未实现：{e}", fg="yellow"))
        sys.exit(1)

    click.echo(
        f"正在获取 {framework} ({version}) API 列表...\n"
        f"（首次运行需拉取多个模块页面，约 1-2 分钟；24h 内命中缓存）",
        err=True,
    )

    use_cache = not no_cache
    try:
        fetched_apis = fetcher.fetch_all_apis(
            framework=framework,
            version=version,
            use_cache=use_cache,
            skip_namespaces=all_skip if all_skip else None,
        )
    except Exception as e:
        click.echo(click.style(f"获取失败：{e}", fg="red"))
        sys.exit(1)

    click.echo(f"共获取到 {len(fetched_apis)} 个 API 条目", err=True)

    if cat is None:
        click.echo(click.style("无法加载算子目录，仅显示原始 API 列表。", fg="yellow"))
        if as_json:
            click.echo(json.dumps(fetched_apis, ensure_ascii=False, indent=2))
        else:
            for api in fetched_apis[:50]:
                click.echo(f"  [{api.get('type','?'):8}] {api['name']:<50}  {api.get('signature','')}")
            if len(fetched_apis) > 50:
                click.echo(f"  ... 共 {len(fetched_apis)} 条，使用 --json 查看全部")
        return

    # 执行 diff
    diff = cat.diff_with_fetched(framework, fetched_apis)

    if as_json:
        click.echo(json.dumps(diff, ensure_ascii=False, indent=2))
        return

    # ── 输出人类可读报告 ──

    _section = lambda title, count, color: click.echo(
        "\n" + click.style(f"══ {title} ({count}) ", bold=True, fg=color) + "═" * max(0, 60 - len(title) - 5)
    )

    # 1. 签名变更
    changed = diff["changed_signature"]
    _section("签名已变更（在目录中，且有记录签名）", len(changed), "yellow")
    if changed:
        for item in changed:
            click.echo(f"  {click.style(item['name'], bold=True)}")
            click.echo(f"    记录签名:  {item['stored_sig']}")
            click.echo(f"    当前签名:  {click.style(item['current_sig'], fg='yellow')}")
    else:
        click.echo("  无变更")

    # 2. 未在文档中找到的目录算子
    not_found = diff["not_found_in_docs"]
    _section("目录中有，文档中未找到（可能已改名或删除）", len(not_found), "red")
    if not_found:
        for item in not_found:
            click.echo(f"  {click.style(item['name'], fg='red')}")
    else:
        click.echo("  无")

    # 3. 新增未归类 API
    new_apis = diff["new_uncategorized"]
    _section("新 API（不在目录也不在排除列表）", len(new_apis), "green")
    if new_apis:
        # 按类型分组显示
        by_type: Dict[str, List] = {}
        for item in new_apis:
            by_type.setdefault(item.get("type", "other"), []).append(item)
        for atype, items in sorted(by_type.items()):
            click.echo(f"  [{atype}]  {len(items)} 个")
            for item in items[:10]:
                click.echo(f"    {item['name']:<55}  {item.get('signature','')}")
            if len(items) > 10:
                click.echo(f"    ... 还有 {len(items) - 10} 个，使用 --json 查看全部")
    else:
        click.echo("  无新增")

    # 4. 无存储签名（可选显示）
    no_sig = diff["no_stored_signature"]
    if show_no_sig:
        _section("目录中有，但未记录签名（供参考）", len(no_sig), "cyan")
        for item in no_sig:
            click.echo(f"  {item['name']:<55}  {item.get('current_sig','')}")
    else:
        click.echo(
            click.style(
                f"\n  提示：目录中有 {len(no_sig)} 个算子尚未记录签名。"
                f"运行加 --show-no-sig 可查看详情；"
                f"手动在 pytorch.yaml 中添加 signature 字段后可追踪签名变化。",
                fg="cyan",
            )
        )

    click.echo("")


# ── import-from-docs ───────────────────────────────────────────────────────────

@catalog.command("import-from-docs")
@click.option(
    "--framework", "-f",
    default="pytorch",
    type=click.Choice(["pytorch"], case_sensitive=False),
    show_default=True,
    help="目标框架（当前支持 pytorch）",
)
@click.option(
    "--version", "-v",
    default="stable",
    show_default=True,
    metavar="VERSION",
    help="文档版本（非 stable 暂未实现）",
)
@click.option(
    "--replace",
    is_flag=True,
    default=False,
    help="清空现有算子目录，完全替换为文档中获取的 API 列表",
)
@click.option("--no-cache", is_flag=True, default=False, help="忽略缓存，强制重新拉取文档")
@click.option("--dry-run", is_flag=True, default=False, help="试运行：显示将要写入的内容，不实际修改文件")
@click.option("--yes", "-y", is_flag=True, default=False, help="跳过确认提示直接执行")
def catalog_import_from_docs(framework, version, replace, no_cache, dry_run, yes):
    """从官方文档批量导入 API 到算子目录（无需 LLM）。

    拉取官方文档中所有 API 条目，经排除列表过滤后，写入算子目录 YAML。

    \b
    模式说明：
      默认（合并模式）：仅添加目录中不存在的新 API，保留已有条目的分类/版本等元数据
      --replace 模式 ：清空现有目录，以文档 API 列表完全重建（元数据将丢失）

    导入的条目：
      - category: 空（待手动标注）
      - since:    空（待手动标注）
      - type:     从命名约定推断（首字母大写 = class，否则 function）

    \b
    示例：
      # 合并模式（仅添加新 API）
      deepmt catalog import-from-docs

      # 替换模式（清空后全量导入）
      deepmt catalog import-from-docs --replace

      # 试运行：查看将导入什么，不写文件
      deepmt catalog import-from-docs --replace --dry-run

      # 跳过确认
      deepmt catalog import-from-docs --replace --yes
    """
    try:
        from deepmt.tools.web_search.api_list_fetcher import APIListFetcher
        from deepmt.mr_generator.base.operator_catalog import OperatorCatalog
        fetcher = APIListFetcher()
        cat = OperatorCatalog()
    except Exception as e:
        click.echo(click.style(f"错误：{e}", fg="red"), err=True)
        sys.exit(1)

    # 版本检查
    try:
        fetcher._check_supported(framework, version)
    except NotImplementedError as e:
        click.echo(click.style(f"未实现：{e}", fg="yellow"))
        sys.exit(1)

    # 加载排除列表
    exclude_cfg = cat.load_exclude_config(framework)
    skip_ns = exclude_cfg.get("excluded_namespaces", [])

    click.echo(
        f"正在获取 {framework} ({version}) API 列表...",
        err=True,
    )
    use_cache = not no_cache
    try:
        fetched_apis = fetcher.fetch_all_apis(
            framework=framework,
            version=version,
            use_cache=use_cache,
            skip_namespaces=skip_ns if skip_ns else None,
        )
    except Exception as e:
        click.echo(click.style(f"获取失败：{e}", fg="red"))
        sys.exit(1)

    # 过滤排除列表
    kept = [api for api in fetched_apis if not cat.is_excluded(api["name"], exclude_cfg)]
    excluded_count = len(fetched_apis) - len(kept)
    click.echo(
        f"共 {len(fetched_apis)} 个 API，排除 {excluded_count} 个后剩余 {len(kept)} 个",
        err=True,
    )

    # 获取现有目录
    existing_entries = cat.get_all_entries(framework)
    existing_names = {e.name for e in existing_entries}
    existing_aliases: set = set()
    for e in existing_entries:
        existing_aliases.update(e.aliases)

    if replace:
        # ── 替换模式：全量重建 ──
        to_import = kept
        new_count = len(to_import)
        preserved_count = 0
        click.echo(
            f"替换模式：将清空 {len(existing_entries)} 个现有条目，导入 {new_count} 个新条目",
            err=True,
        )
    else:
        # ── 合并模式：仅添加新 API ──
        to_import_new = [
            api for api in kept
            if api["name"] not in existing_names and api["name"] not in existing_aliases
        ]
        new_count = len(to_import_new)
        preserved_count = len(existing_entries)
        click.echo(
            f"合并模式：保留 {preserved_count} 个现有条目，新增 {new_count} 个",
            err=True,
        )
        to_import = to_import_new

    if new_count == 0 and not replace:
        click.echo(click.style("无新增 API，目录已是最新。", fg="green"))
        return

    # 显示样本
    click.echo(f"\n将导入以下 API（共 {new_count} 个，前 20 条）：")
    by_type: Dict[str, int] = {}
    for api in to_import:
        by_type[api.get("type", "?")] = by_type.get(api.get("type", "?"), 0) + 1
    for t, cnt in sorted(by_type.items()):
        click.echo(f"  [{t}]  {cnt} 个")
    click.echo()
    for api in to_import[:20]:
        click.echo(f"  {api['name']:<60}  [{api.get('type', '?')}]")
    if len(to_import) > 20:
        click.echo(f"  ... 还有 {len(to_import) - 20} 个")

    if dry_run:
        click.echo(click.style("\n[dry-run] 未写入任何文件。", fg="yellow"))
        return

    # 确认
    if not yes:
        action_desc = "清空目录并全量导入" if replace else "添加新 API"
        if not click.confirm(f"\n确认{action_desc}？"):
            click.echo("已取消。")
            return

    # ── 执行写入 ──
    import yaml as _yaml
    from pathlib import Path as _Path
    from datetime import datetime as _dt

    catalog_dir = _Path(__file__).parent.parent.parent / "mr_generator" / "config" / "operator_catalog"
    yaml_path = catalog_dir / f"{framework}.yaml"

    # 读取现有文件头（注释行 + 非 operators 的字段行）
    header_lines: List[str] = []
    if yaml_path.exists():
        with yaml_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("operators:"):
                    break
                if line.startswith("last_updated:"):
                    continue
                header_lines.append(line.rstrip())

    today = _dt.now().strftime("%Y-%m-%d")

    # 构造最终条目列表
    if replace:
        # 全量替换：用 fetched 列表重建，但保留原有条目的元数据（如果名称匹配）
        existing_by_name = {e.name: e for e in existing_entries}
        final_entries = []
        for api in to_import:
            name = api["name"]
            if name in existing_by_name:
                # 保留已有元数据
                final_entries.append(existing_by_name[name])
            else:
                final_entries.append({
                    "name": name,
                    "type": api.get("type", ""),
                    "url": api.get("url", ""),
                })
    else:
        # 合并：保留现有条目，追加新条目
        final_entries = list(existing_entries)  # existing OperatorEntry objects
        for api in to_import:
            final_entries.append({
                "name": api["name"],
                "type": api.get("type", ""),
                "url": api.get("url", ""),
            })

    # 写入 YAML
    with yaml_path.open("w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        f.write(f'last_updated: "{today}"\n\n')
        f.write("operators:\n")
        for entry in final_entries:
            if hasattr(entry, "to_dict"):
                # 现有 OperatorEntry 对象
                d = entry.to_dict()
                f.write(f"  - name: {d['name']}\n")
                if d.get("category"):
                    f.write(f"    category: {d['category']}\n")
                if d.get("since") and d["since"] != "0.0":
                    f.write(f"    since: \"{d['since']}\"\n")
                if d.get("deprecated"):
                    f.write(f"    deprecated: \"{d['deprecated']}\"\n")
                if d.get("removed"):
                    f.write(f"    removed: \"{d['removed']}\"\n")
                if d.get("aliases"):
                    aliases_str = "[" + ", ".join(d["aliases"]) + "]"
                    f.write(f"    aliases: {aliases_str}\n")
                if d.get("doc_url"):
                    f.write(f"    doc_url: {d['doc_url']}\n")
                if d.get("signature"):
                    f.write(f"    signature: \"{d['signature']}\"\n")
                if d.get("note"):
                    f.write(f"    note: \"{d['note']}\"\n")
            else:
                # 新导入的 dict 条目
                f.write(f"  - name: {entry['name']}\n")
                if entry.get("type"):
                    f.write(f"    api_type: {entry['type']}\n")
                if entry.get("url"):
                    f.write(f"    doc_url: {entry['url']}\n")
            f.write("\n")

    click.echo(click.style(
        f"\n✓ 已写入 {len(final_entries)} 个条目到 {yaml_path}",
        fg="green", bold=True,
    ))
