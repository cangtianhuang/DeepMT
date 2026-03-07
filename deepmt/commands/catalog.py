"""
deepmt catalog — 算子目录管理子命令组

命令:
    list    列出指定框架的算子（支持按分类筛选）
    search  跨框架模糊搜索算子
    info    查询某算子的跨框架分布及知识库 MR 数量
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
