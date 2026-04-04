"""
deepmt catalog — 算子目录管理子命令组

命令:
    list             列出指定框架的算子（支持按分类筛选）
    search           跨框架模糊搜索算子
    info             查询某算子的跨框架分布及知识库 MR 数量
    sync             通过 CrawlAgent 自动更新算子目录
    latest-version   快速获取框架最新稳定版本号（无需 LLM）
    fetch-doc        获取并打印算子文档正文
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
        from mr_generator.base.mr_repository import MRRepository
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


# ── sync ───────────────────────────────────────────────────────────────────────

@catalog.command("sync")
@click.option(
    "--framework", "-f",
    default=None,
    type=click.Choice(_ALL_FRAMEWORKS, case_sensitive=False),
    help="目标框架（不指定则同步所有框架）",
)
@click.option(
    "--version", "-v",
    default="latest",
    show_default=True,
    metavar="VERSION",
    help="框架版本号（如 2.5.1），默认 latest",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="忽略缓存，强制重新爬取",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="试运行：仅打印将写入的内容，不修改 YAML",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    default=False,
    help="静默模式：不打印 Agent 每步执行过程",
)
def catalog_sync(framework, version, no_cache, dry_run, quiet):
    """通过 CrawlAgent 自动爬取并更新算子目录 YAML。

    需要配置 LLM API Key。

    \b
    示例:
      deepmt catalog sync                              # 同步所有框架
      deepmt catalog sync --framework pytorch          # 仅同步 PyTorch
      deepmt catalog sync --framework pytorch --no-cache   # 忽略缓存重新爬取
      deepmt catalog sync --framework pytorch --dry-run    # 试运行
    """
    try:
        from tools.agent.task_runner import TaskRunner
        runner = TaskRunner(verbose=not quiet)
    except Exception as e:
        click.echo(click.style(f"错误：无法初始化 TaskRunner：{e}", fg="red"))
        sys.exit(1)

    frameworks_to_sync = [framework] if framework else _ALL_FRAMEWORKS
    use_cache = not no_cache
    save_yaml = not dry_run

    total_added = 0
    total_updated = 0
    errors: List = []

    for fw in frameworks_to_sync:
        click.echo(f"\n正在同步 {click.style(fw, bold=True)}（version={version}）…")
        try:
            stats = runner.sync_catalog(fw, version=version, use_cache=use_cache, save_yaml=save_yaml)
            added = stats.get("added", 0)
            updated = stats.get("updated", 0)
            skipped = stats.get("skipped", 0)
            total_added += added
            total_updated += updated

            status_parts = []
            if added:
                status_parts.append(click.style(f"+{added} 新增", fg="green"))
            if updated:
                status_parts.append(click.style(f"~{updated} 更新", fg="cyan"))
            if skipped:
                status_parts.append(f"{skipped} 跳过")
            status_str = "  ".join(status_parts) if status_parts else "无变化"

            prefix = click.style("✓", fg="green") if not dry_run else click.style("~", fg="yellow")
            dry_tag = "  [dry-run，未写入]" if dry_run else ""
            click.echo(f"  {prefix} {fw}: {status_str}{dry_tag}")

        except Exception as e:
            errors.append((fw, str(e)))
            click.echo(f"  {click.style('✗', fg='red')} {fw}: {e}")

    # 汇总
    click.echo("")
    if not errors:
        if dry_run:
            click.echo(click.style("试运行完成，未写入任何文件。", fg="yellow"))
        else:
            click.echo(click.style(
                f"同步完成：{len(frameworks_to_sync)} 个框架，新增 {total_added}，更新 {total_updated}。",
                fg="green",
            ))
    else:
        click.echo(click.style(
            f"同步完成（有 {len(errors)} 个框架失败）：新增 {total_added}，更新 {total_updated}。",
            fg="yellow",
        ))
        sys.exit(1)


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
        from tools.web_search.search_agent import SearchAgent
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
        from tools.web_search.search_agent import SearchAgent
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
@click.option("--no-cache", is_flag=True, default=False, help="忽略本地缓存，强制重新拉取")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出 API 列表")
@click.option("--show-cache-path", is_flag=True, default=False, help="显示缓存文件路径")
def catalog_update_api_list(framework, no_cache, as_json, show_cache_path):
    """从官方文档获取 API 模块列表并缓存到本地（无需 LLM）。

    从 pytorch-api.html 等页面解析所有 API 模块条目（名称 + 链接），
    保存至 tools/web_search/.cache/ 目录，后续查询直接命中缓存。

    \b
    示例:
      deepmt catalog update-api-list
      deepmt catalog update-api-list --no-cache
      deepmt catalog update-api-list --json
      deepmt catalog update-api-list --show-cache-path
    """
    from tools.web_search.search_agent import _FRAMEWORK_API_PAGES
    api_url = _FRAMEWORK_API_PAGES.get(framework)
    if not api_url:
        click.echo(click.style(f"框架 '{framework}' 暂不支持 API 列表更新。", fg="yellow"))
        sys.exit(1)

    try:
        from tools.web_search.search_agent import SearchAgent
        agent = SearchAgent()
    except Exception as e:
        click.echo(click.style(f"错误：{e}", fg="red"), err=True)
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
    click.echo(f"正在{action}拉取 {framework} API 列表: {api_url}", err=True)

    entries = agent.fetch_api_list(api_url, use_cache=use_cache)

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
