"""
DeepMT Dashboard — JSON API 端点

所有端点返回统一结构：
    { "data": ..., "generated_at": "ISO8601", "error": null }

缓存策略（in-process TTL，演示场景）：
    /api/summary          — 30s（ExperimentOrganizer 读文件成本较高）
    /api/mr-repository    — 60s（YAML 遍历）
    /api/test-results     — 10s（SQLite 查询快，允许近实时）
    /api/cross-framework  — 60s（JSON 文件列表）
    /api/evidence         — 30s
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter

from deepmt import __version__
from deepmt.analysis.evidence_collector import EvidenceCollector
from deepmt.analysis.cross_framework_tester import CrossFrameworkTester
from deepmt.analysis.experiment_organizer import ExperimentOrganizer
from deepmt.core.results_manager import ResultsManager
from deepmt.mr_generator.base.mr_repository import MRRepository

router = APIRouter(tags=["API"])

# ── 通用工具 ───────────────────────────────────────────────────────────────────

_CACHE: Dict[str, Any] = {}  # { key: (data, timestamp) }


def _cached(key: str, ttl: int, factory):
    """简单 TTL 缓存，避免每次请求都重新读取文件/数据库。"""
    entry = _CACHE.get(key)
    if entry:
        data, ts = entry
        if time.time() - ts < ttl:
            return data
    data = factory()
    _CACHE[key] = (data, time.time())
    return data


def _ok(data: Any) -> Dict:
    return {
        "data": data,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "error": None,
    }


def _err(msg: str) -> Dict:
    return {
        "data": None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "error": msg,
    }


# ── 健康检查 ───────────────────────────────────────────────────────────────────


@router.get("/health", summary="服务健康检查")
async def api_health():
    """返回服务状态与版本信息。"""
    return _ok({"status": "ok", "version": __version__})


# ── F11-6: RQ1-RQ4 摘要 ────────────────────────────────────────────────────────


@router.get("/summary", summary="RQ1-RQ4 全量摘要")
async def api_summary():
    """
    调用 ExperimentOrganizer.collect_all() 返回 RQ1-RQ4 结构化数据。
    缓存 30 秒。
    """
    def _load():
        try:
            return ExperimentOrganizer().collect_all()
        except Exception as e:
            return {"error": str(e)}

    data = _cached("summary", 30, _load)
    return _ok(data)


# ── F11-7: MR 仓库 ─────────────────────────────────────────────────────────────


@router.get("/mr-repository", summary="MR 仓库列表与统计")
async def api_mr_repository():
    """
    遍历 MR 仓库，返回每个算子的 MR 统计（数量、验证率、类别、来源）
    以及全局分布数据。缓存 60 秒。
    """
    def _load():
        try:
            repo = MRRepository()
            operators_raw = repo.list_operators()

            total_mrs = 0
            verified_mrs = 0
            category_dist: Dict[str, int] = {}
            source_dist: Dict[str, int] = {}
            operators_out = []

            for op_name in operators_raw:
                mrs = repo.load(op_name)
                if not mrs:
                    continue

                op_total = len(mrs)
                op_verified = sum(1 for m in mrs if m.verified)
                total_mrs += op_total
                verified_mrs += op_verified

                cats = set()
                srcs = set()
                fws: set = set()
                for mr in mrs:
                    cat = mr.category or "uncategorized"
                    src = mr.source or "unknown"
                    category_dist[cat] = category_dist.get(cat, 0) + 1
                    source_dist[src] = source_dist.get(src, 0) + 1
                    cats.add(cat)
                    srcs.add(src)
                    if mr.applicable_frameworks:
                        fws.update(mr.applicable_frameworks)

                operators_out.append({
                    "operator_name": op_name,
                    "total_count": op_total,
                    "verified_count": op_verified,
                    "categories": sorted(cats),
                    "sources": sorted(srcs),
                    "frameworks": sorted(fws) if fws else ["pytorch"],
                })

            n_ops = len(operators_out)
            return {
                "total_mr_count": total_mrs,
                "verified_mr_count": verified_mrs,
                "verification_rate": round(verified_mrs / total_mrs, 4) if total_mrs else 0.0,
                "operators_with_mr": n_ops,
                "avg_mr_per_operator": round(total_mrs / n_ops, 2) if n_ops else 0.0,
                "category_distribution": dict(
                    sorted(category_dist.items(), key=lambda x: -x[1])
                ),
                "source_distribution": dict(
                    sorted(source_dist.items(), key=lambda x: -x[1])
                ),
                "operators": operators_out,
            }
        except Exception as e:
            return {"error": str(e), "operators": []}

    return _ok(_cached("mr_repository", 60, _load))


@router.get("/mr-repository/{operator_name:path}", summary="单算子 MR 列表")
async def api_mr_detail(operator_name: str):
    """
    返回指定算子的全部 MR，含 transform_code / oracle_expr / 验证状态等。
    不缓存（单算子查询成本低）。
    """
    try:
        repo = MRRepository()
        mrs = repo.load(operator_name)
        data = [
            {
                "id": mr.id,
                "description": mr.description,
                "transform_code": mr.transform_code,
                "oracle_expr": mr.oracle_expr,
                "category": mr.category,
                "source": mr.source,
                "tolerance": mr.tolerance,
                "layer": mr.layer,
                "applicable_frameworks": mr.applicable_frameworks,
                "verified": mr.verified,
                "checked": mr.checked,
                "proven": mr.proven,
                "analysis": mr.analysis,
            }
            for mr in mrs
        ]
        return _ok(data)
    except Exception as e:
        return _err(str(e))


# ── F11-8: 测试结果 ─────────────────────────────────────────────────────────────


@router.get("/test-results", summary="各算子测试结果汇总")
async def api_test_results():
    """
    读取 defect_summary 表，按 (ir_name, framework) 聚合取最新一条。
    缓存 10 秒。
    """
    def _load():
        try:
            rows = ResultsManager().get_summary()
            # defect_summary 可能有重复 (ir_name, framework) 行；取每组最新的
            latest: Dict[str, Any] = {}
            for r in rows:
                key = f"{r.get('ir_name')}|{r.get('framework', 'pytorch')}"
                if key not in latest or r.get("last_updated", "") > latest[key].get("last_updated", ""):
                    latest[key] = r
            # 按失败数降序，突出问题算子
            return sorted(latest.values(), key=lambda r: r.get("failed_tests", 0), reverse=True)
        except Exception as e:
            return []

    return _ok(_cached("test_results", 10, _load))


@router.get("/test-results/failed", summary="近期失败用例列表")
async def api_test_failed(limit: int = 50):
    """
    读取 test_results 表中的 FAIL 记录，按时间倒序。
    不缓存（便于快速反映最新失败）。
    """
    try:
        rows = ResultsManager().get_failed_tests(limit=limit)
        return _ok(rows)
    except Exception as e:
        return _ok([])


# ── F11-9: 证据包 ──────────────────────────────────────────────────────────────


@router.get("/evidence", summary="证据包列表")
async def api_evidence(limit: int = 100):
    """
    列出所有证据包的摘要（不含 reproduce_script，节省传输）。
    缓存 30 秒。
    """
    def _load():
        try:
            packs = EvidenceCollector().list_all(limit=limit)
            return [
                {
                    "id": p.evidence_id,
                    "operator": p.operator,
                    "framework": p.framework,
                    "framework_version": p.framework_version,
                    "mr_id": p.mr_id,
                    "mr_description": p.mr_description,
                    "actual_diff": p.actual_diff,
                    "tolerance": p.tolerance,
                    "detail": p.detail,
                    "timestamp": p.timestamp,
                }
                for p in packs
            ]
        except Exception as e:
            return []

    return _ok(_cached("evidence", 30, _load))


@router.get("/evidence/{evidence_id}/script", summary="证据包复现脚本")
async def api_evidence_script(evidence_id: str):
    """
    返回指定证据包的完整 Python 复现脚本。
    不缓存（按需访问）。
    """
    try:
        pack = EvidenceCollector().load(evidence_id)
        if pack is None:
            return _err(f"证据包 '{evidence_id}' 不存在")
        return _ok({
            "id": pack.evidence_id,
            "operator": pack.operator,
            "mr_description": pack.mr_description,
            "script": pack.reproduce_script,
            "timestamp": pack.timestamp,
        })
    except Exception as e:
        return _err(str(e))


# ── F11-10: 跨框架一致性 ────────────────────────────────────────────────────────


@router.get("/cross-framework", summary="跨框架实验会话列表")
async def api_cross_framework():
    """
    加载 data/results/cross_framework/ 下所有会话 JSON，返回摘要列表（不含 mr_results 详情）。
    缓存 60 秒。
    """
    def _load():
        try:
            sessions = CrossFrameworkTester().load_all()
            return [
                {
                    "session_id": s.session_id,
                    "operator": s.operator,
                    "framework1": s.framework1,
                    "framework2": s.framework2,
                    "n_samples": s.n_samples,
                    "mr_count": s.mr_count,
                    "overall_consistency_rate": round(s.overall_consistency_rate, 4),
                    "output_max_diff": (
                        s.output_max_diff
                        if s.output_max_diff == s.output_max_diff  # NaN 判断
                        else None
                    ),
                    "inconsistent_mr_count": s.inconsistent_mr_count,
                    "timestamp": s.timestamp,
                }
                for s in sessions
            ]
        except Exception as e:
            return []

    return _ok(_cached("cross_framework", 60, _load))


@router.get("/cross-framework/{session_id}", summary="跨框架会话详情")
async def api_cross_session(session_id: str):
    """
    返回指定会话的完整数据（含每条 MR 的一致性对比结果）。
    不缓存（按需访问）。
    """
    try:
        sessions = CrossFrameworkTester().load_all()
        session = next((s for s in sessions if s.session_id == session_id), None)
        if session is None:
            return _err(f"会话 '{session_id}' 不存在")

        data = session.to_dict()
        # 补充每条 MR 的通过率字段（to_dict 已包含，但做二次确认）
        return _ok(data)
    except Exception as e:
        return _err(str(e))
