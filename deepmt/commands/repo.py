"""
deepmt repo — MR 知识库管理子命令组

命令:
    list    列出知识库中所有主体（算子/模型/应用）
    stats   显示统计信息（含质量等级分布）
    info    显示主体详情（MR 数量等）
    delete  删除 MR 记录
    retire  归档（退役）指定 MR
    filter  按质量/层次/框架筛选并展示 MR
    audit   运行全库审计并输出质量报告
"""

import json
import sys

import click

from deepmt._utils import get_repo

_LAYERS = ("operator", "model", "application")
_QUALITY_LEVELS = ("candidate", "checked", "proven", "curated", "retired")


@click.group()
def repo():
    """MR 知识库管理（查看、统计、审计、治理）。"""


# ── list ──────────────────────────────────────────────────────────────────────

@repo.command("list")
@click.option("--layer", default="operator", show_default=True,
              type=click.Choice(_LAYERS), help="MR 层次")
@click.option("--framework", default=None, help="按框架过滤（如 pytorch）")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def repo_list(layer, framework, as_json):
    """列出知识库中所有主体（算子/模型/应用）。

    \b
    示例:
      deepmt repo list
      deepmt repo list --layer model
      deepmt repo list --framework pytorch
      deepmt repo list --json
    """
    r = get_repo(layer=layer)
    if framework:
        subjects = r.list_subjects_by_framework(framework)
    else:
        subjects = r.list_subjects()

    if as_json:
        result = []
        for subj in subjects:
            stats = r.get_statistics(subj)
            result.append({"subject": subj, "layer": layer, **stats})
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if not subjects:
        click.echo(f"[{layer}] 知识库为空，尚未保存任何 MR。")
        return

    header = f"[{layer}] 共有 {len(subjects)} 个主体"
    if framework:
        header += f"（框架: {framework}）"
    click.echo(header + ":\n")
    click.echo(f"  {'主体':<40} {'总数':>6} {'已验证':>8} {'已退役':>8}")
    click.echo("  " + "─" * 66)
    for subj in subjects:
        s = r.get_statistics(subj)
        click.echo(
            f"  {subj:<40} {s['total_mrs']:>6} {s['verified_mrs']:>8} {s['retired']:>8}"
        )


# ── stats ─────────────────────────────────────────────────────────────────────

@repo.command("stats")
@click.option("--layer", default="operator", show_default=True,
              type=click.Choice(_LAYERS), help="MR 层次")
@click.option("--all-layers", "all_layers", is_flag=True, default=False,
              help="汇总全部三层统计")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def repo_stats(layer, all_layers, as_json):
    """显示知识库整体统计信息（含质量等级分布）。

    \b
    示例:
      deepmt repo stats
      deepmt repo stats --layer model
      deepmt repo stats --all-layers
      deepmt repo stats --json
    """
    if all_layers:
        result = {}
        for lyr in _LAYERS:
            r = get_repo(layer=lyr)
            result[lyr] = r.get_statistics()
        if as_json:
            click.echo(json.dumps(result, ensure_ascii=False, indent=2))
            return
        for lyr, stats in result.items():
            _print_layer_stats(lyr, stats)
        return

    r = get_repo(layer=layer)
    stats = r.get_statistics()

    if as_json:
        click.echo(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    _print_layer_stats(layer, stats)


def _print_layer_stats(layer: str, stats: dict):
    click.echo(f"\n[{layer}] MR 知识库统计")
    click.echo("─" * 40)
    click.echo(f"  总 MR 数:   {stats['total_mrs']}")
    click.echo(f"  已验证:     {stats['verified_mrs']}")
    click.echo(f"  未验证:     {stats['unverified_mrs']}")
    click.echo(f"  数值检查:   {stats['checked']}")
    click.echo(f"  符号证明:   {stats['proven']}")
    click.echo(f"  已退役:     {stats['retired']}")

    qd = stats.get("quality_dist", {})
    if qd:
        click.echo("\n  质量等级分布:")
        for lvl in ["curated", "proven", "checked", "candidate", "retired"]:
            cnt = qd.get(lvl, 0)
            if cnt:
                bar = "█" * min(cnt, 20)
                click.echo(f"    {lvl:<12} {cnt:>4}  {bar}")

    sd = stats.get("by_source", {})
    if sd:
        click.echo("\n  来源分布:")
        for src, cnt in sorted(sd.items(), key=lambda x: -x[1]):
            click.echo(f"    {src:<16} {cnt:>4}")

    by_subj = stats.get("by_subject", {})
    if by_subj:
        click.echo(f"\n  主体分布 ({len(by_subj)} 个):")
        click.echo(f"  {'主体':<40} {'总数':>6} {'已验证':>8} {'已退役':>8}")
        click.echo("  " + "─" * 66)
        for subj, s in by_subj.items():
            click.echo(
                f"  {subj:<40} {s['total']:>6} {s['verified']:>8} {s.get('retired', 0):>8}"
            )


# ── info ──────────────────────────────────────────────────────────────────────

@repo.command("info")
@click.argument("subject")
@click.option("--layer", default="operator", show_default=True,
              type=click.Choice(_LAYERS), help="MR 层次")
@click.option("--framework", default=None, help="按框架过滤 MR（如 pytorch）")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def repo_info(subject, layer, framework, as_json):
    """显示主体的详细信息（MR 摘要）。

    \b
    示例:
      deepmt repo info relu
      deepmt repo info relu --layer operator --framework pytorch
      deepmt repo info ResNet50 --layer model --json
    """
    r = get_repo(layer=layer)

    if not r.exists(subject):
        click.echo(click.style(f"[{layer}] 知识库中未找到主体 '{subject}'", fg="yellow"))
        sys.exit(1)

    stats = r.get_statistics(subject)

    if as_json:
        mrs = r.get_mr_with_validation_status(subject, framework=framework)
        result = {
            "subject": subject,
            "layer": layer,
            "stats": stats,
            "mrs": [
                {
                    "id": m.id,
                    "description": m.description,
                    "category": m.category,
                    "oracle_expr": m.oracle_expr,
                    "transform_code": m.transform_code,
                    "lifecycle_state": m.lifecycle_state,
                    "quality_level": m.quality_level,
                    "verified": m.verified,
                    "source": m.source,
                    "applicable_frameworks": m.applicable_frameworks,
                    "provenance": m.provenance,
                }
                for m in mrs
            ],
        }
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return

    click.echo(f"\n[{layer}] 主体: {click.style(subject, bold=True)}")
    click.echo(f"  总 MR 数:  {stats['total_mrs']}")
    click.echo(f"  已验证:    {stats['verified_mrs']}")
    click.echo(f"  已退役:    {stats['retired']}")

    mrs = r.get_mr_with_validation_status(subject, framework=framework)
    fw_note = f"(框架: {framework})" if framework else ""
    click.echo(f"\n  {fw_note}共 {len(mrs)} 个 MR:")
    for m in mrs:
        if m.lifecycle_state == "retired":
            mark = click.style("⊘", fg="bright_black")
        elif m.verified:
            mark = click.style("✓", fg="green")
        else:
            mark = click.style("✗", fg="red")
        fw_tag = f" [{'/'.join(m.applicable_frameworks)}]" if m.applicable_frameworks else ""
        ql_tag = click.style(f"[{m.quality_level}]", fg=_quality_color(m.quality_level))
        click.echo(f"    [{mark}] {ql_tag} [{m.category}]{fw_tag} {m.description}")
        click.echo(f"         oracle: {m.oracle_expr}")
        if m.provenance:
            gen = m.provenance.get("generator_id", "")
            ts = m.provenance.get("created_at", "")[:10]
            click.echo(f"         provenance: {gen} @ {ts}")


def _quality_color(quality_level: str) -> str:
    return {
        "curated": "bright_green",
        "proven": "green",
        "checked": "yellow",
        "candidate": "white",
        "retired": "bright_black",
    }.get(quality_level, "white")


# ── delete ────────────────────────────────────────────────────────────────────

@repo.command("delete")
@click.argument("subject")
@click.option("--layer", default="operator", show_default=True,
              type=click.Choice(_LAYERS), help="MR 层次")
@click.option("--id", "mr_id", default=None, help="只删除该 MR ID")
@click.option("--all", "delete_all", is_flag=True, default=False, help="删除主体的全部 MR")
@click.option("--yes", "-y", is_flag=True, default=False, help="跳过确认提示")
def repo_delete(subject, layer, mr_id, delete_all, yes):
    """删除知识库中的 MR 记录。

    \b
    示例:
      deepmt repo delete relu --id <MR_ID>
      deepmt repo delete relu --all
      deepmt repo delete relu --all --yes
    """
    r = get_repo(layer=layer)

    if not r.exists(subject):
        click.echo(click.style(f"[{layer}] 知识库中未找到主体 '{subject}'", fg="yellow"))
        sys.exit(1)

    if not mr_id and not delete_all:
        click.echo(click.style("请指定 --id <MR_ID> 或 --all", fg="red"))
        sys.exit(1)

    desc = f"MR ID={mr_id}（主体 '{subject}'）" if mr_id else f"主体 '{subject}' 的全部 MR"

    if not yes:
        confirmed = click.confirm(f"确认删除 [{layer}] {desc}？", default=False)
        if not confirmed:
            click.echo("已取消。")
            return

    deleted = r.delete(subject, mr_id=mr_id)
    click.echo(click.style(f"已删除 {deleted} 条 MR。", fg="green"))


# ── retire ────────────────────────────────────────────────────────────────────

@repo.command("retire")
@click.argument("subject")
@click.option("--id", "mr_id", required=True, help="要退役的 MR ID")
@click.option("--layer", default="operator", show_default=True,
              type=click.Choice(_LAYERS), help="MR 层次")
@click.option("--yes", "-y", is_flag=True, default=False, help="跳过确认提示")
def repo_retire(subject, mr_id, layer, yes):
    """归档（退役）指定 MR，保留历史记录但不参与测试。

    \b
    示例:
      deepmt repo retire relu --id <MR_ID>
      deepmt repo retire relu --id <MR_ID> --layer operator --yes
    """
    r = get_repo(layer=layer)

    if not r.exists(subject):
        click.echo(click.style(f"[{layer}] 知识库中未找到主体 '{subject}'", fg="yellow"))
        sys.exit(1)

    if not yes:
        confirmed = click.confirm(
            f"确认退役 [{layer}] '{subject}' 中 MR ID={mr_id}？", default=False
        )
        if not confirmed:
            click.echo("已取消。")
            return

    success = r.retire(subject, mr_id)
    if success:
        click.echo(click.style(f"✓ 已退役 MR {mr_id}（保留历史记录）。", fg="green"))
    else:
        click.echo(click.style(f"未找到 MR ID={mr_id}。", fg="yellow"))
        sys.exit(1)


# ── filter ────────────────────────────────────────────────────────────────────

@repo.command("filter")
@click.option("--layer", default=None, type=click.Choice(_LAYERS),
              help="按层次过滤（不指定则全部层）")
@click.option("--min-quality", "min_quality", default="checked", show_default=True,
              type=click.Choice(_QUALITY_LEVELS), help="最低质量等级")
@click.option("--framework", default=None, help="按框架过滤（如 pytorch）")
@click.option("--exclude-retired", "exclude_retired", is_flag=True, default=True,
              help="排除已退役 MR（默认排除）")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
def repo_filter(layer, min_quality, framework, exclude_retired, as_json):
    """按质量/层次/框架筛选 MR，输出满足条件的关系列表。

    \b
    示例:
      deepmt repo filter --min-quality proven
      deepmt repo filter --layer operator --min-quality checked --framework pytorch
      deepmt repo filter --min-quality curated --json
    """
    from deepmt.mr_governance.quality import QualityLevel, filter_by_quality

    target_layers = [layer] if layer else list(_LAYERS)
    min_ql = QualityLevel.from_lifecycle(
        {"candidate": "pending", "checked": "checked",
         "proven": "proven", "curated": "curated", "retired": "retired"}.get(min_quality, "pending")
    )

    all_results = []
    for lyr in target_layers:
        r = get_repo(layer=lyr)
        for subject in r.list_subjects():
            mrs = r.load(subject, framework=framework)
            filtered = filter_by_quality(mrs, min_quality=min_ql, exclude_retired=exclude_retired)
            for m in filtered:
                all_results.append({
                    "layer": lyr,
                    "subject": subject,
                    "id": m.id,
                    "description": m.description,
                    "oracle_expr": m.oracle_expr,
                    "quality_level": m.quality_level,
                    "lifecycle_state": m.lifecycle_state,
                    "source": m.source,
                    "applicable_frameworks": m.applicable_frameworks,
                })

    if as_json:
        click.echo(json.dumps(all_results, ensure_ascii=False, indent=2))
        return

    if not all_results:
        click.echo(f"未找到满足条件的 MR（min_quality={min_quality}）。")
        return

    click.echo(f"\n共 {len(all_results)} 条满足条件的 MR (min_quality={min_quality}):\n")
    click.echo(f"  {'层次':<12} {'主体':<30} {'质量':>10}  描述")
    click.echo("  " + "─" * 80)
    for item in all_results:
        ql_str = click.style(item["quality_level"], fg=_quality_color(item["quality_level"]))
        desc = item["description"][:40]
        click.echo(
            f"  {item['layer']:<12} {item['subject']:<30} {item['quality_level']:>10}  {desc}"
        )


# ── audit ─────────────────────────────────────────────────────────────────────

@repo.command("audit")
@click.option("--layer", default=None, type=click.Choice(_LAYERS),
              help="只审计指定层（默认全部）")
@click.option("--json", "as_json", is_flag=True, default=False, help="以 JSON 格式输出")
@click.option("--pending-review", "pending_review", is_flag=True, default=False,
              help="输出待复核清单（质量低于 proven 的 MR）")
def repo_audit(layer, as_json, pending_review):
    """运行全库审计，输出质量统计与异常报告。

    \b
    示例:
      deepmt repo audit
      deepmt repo audit --layer operator
      deepmt repo audit --json
      deepmt repo audit --pending-review
    """
    from deepmt.analysis.repo_audit import RepoAuditor
    from deepmt.mr_governance.quality import QualityLevel

    auditor = RepoAuditor()
    target_layers = [layer] if layer else None
    report = auditor.run_audit(layers=target_layers)

    if pending_review:
        text = auditor.export_pending_review(
            min_quality=QualityLevel.PROVEN,
            output_format="json" if as_json else "text",
        )
        click.echo(text)
        return

    if as_json:
        data = {
            "total_mrs": report.total_mrs,
            "total_retired": report.total_retired,
            "total_duplicate_groups": report.total_duplicate_groups,
            "quality_distribution": report.quality_distribution(),
            "source_distribution": report.source_distribution(),
            "anomalies": report.anomalies(),
            "layers": {
                lyr: {
                    "total": s.total,
                    "by_quality": s.by_quality,
                    "by_source": s.by_source,
                    "retired": s.retired,
                    "no_oracle": s.no_oracle,
                    "no_provenance": s.no_provenance,
                    "subjects": s.subjects,
                    "duplicate_groups": [
                        {
                            "canonical_id": g.canonical_id,
                            "duplicate_ids": g.duplicate_ids,
                            "similarity_type": g.similarity_type,
                            "reason": g.reason,
                        }
                        for g in s.duplicate_groups
                    ],
                }
                for lyr, s in report.layers.items()
            },
        }
        click.echo(json.dumps(data, ensure_ascii=False, indent=2))
        return

    click.echo(report.summary_text())
