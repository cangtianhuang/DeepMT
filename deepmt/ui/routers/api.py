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

import asyncio
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter

from deepmt import __version__
from deepmt.analysis.reporting.evidence_collector import EvidenceCollector
from deepmt.analysis.qa.cross_framework_tester import CrossFrameworkTester
from deepmt.analysis.qa.repo_audit import RepoAuditor
from deepmt.experiments.case_study import CaseStudyIndex
from deepmt.experiments.organizer import ExperimentOrganizer
from deepmt.core.results_manager import ResultsManager
from deepmt.mr_generator.base.mr_repository import MRRepository
from deepmt.mr_governance.quality import QualityLevel, filter_by_quality

router = APIRouter(tags=["API"])

# ── 通用工具 ───────────────────────────────────────────────────────────────────

_CACHE: Dict[str, Any] = {}             # { key: (data, timestamp) }
_CACHE_LOCKS: Dict[str, threading.Lock] = {}
_CACHE_META_LOCK = threading.Lock()


def _get_cache_lock(key: str) -> threading.Lock:
    with _CACHE_META_LOCK:
        if key not in _CACHE_LOCKS:
            _CACHE_LOCKS[key] = threading.Lock()
        return _CACHE_LOCKS[key]


def _cached(key: str, ttl: int, factory):
    """
    线程安全的 TTL 缓存。
    每个 key 持有独立锁（double-checked locking），防止缓存穿透与并发重建。
    """
    entry = _CACHE.get(key)
    if entry:
        data, ts = entry
        if time.time() - ts < ttl:
            return data
    lock = _get_cache_lock(key)
    with lock:
        # double-check：持锁后再次验证，另一线程可能已填充
        entry = _CACHE.get(key)
        if entry:
            data, ts = entry
            if time.time() - ts < ttl:
                return data
        data = factory()
        _CACHE[key] = (data, time.time())
        return data


async def _acached(key: str, ttl: int, factory):
    """
    异步 TTL 缓存包装器。
    在 FastAPI 的异步事件循环中，通过 asyncio.to_thread 将阻塞 I/O 隔离到
    线程池，避免占用事件循环导致界面卡顿。
    """
    # 快速路径：已命中缓存时直接在事件循环中返回，避免线程调度开销
    entry = _CACHE.get(key)
    if entry:
        data, ts = entry
        if time.time() - ts < ttl:
            return data
    return await asyncio.to_thread(_cached, key, ttl, factory)


def _sanitize(obj: Any) -> Any:
    """递归将 NaN/Inf 浮点替换为 None，确保 JSON 序列化安全。"""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _ok(data: Any) -> Dict:
    return {
        "data": _sanitize(data),
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

    data = await _acached("summary", 30, _load)
    return _ok(data)


# ── F11-7: MR 仓库 ─────────────────────────────────────────────────────────────


@router.get("/mr-repository", summary="MR 仓库列表与统计")
async def api_mr_repository(layer: str = "operator"):
    """
    遍历 MR 仓库，返回每个主体（算子/模型/应用）的 MR 统计。
    layer 参数：operator（默认）/ model / application。缓存 60 秒。
    """
    valid_layers = {"operator", "model", "application"}
    if layer not in valid_layers:
        return _err(f"layer 必须是 {valid_layers} 之一")

    cache_key = f"mr_repository_{layer}"

    def _load():
        try:
            repo = MRRepository(layer=layer)
            operators_raw = repo.list_subjects()

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
                    "subject_name": op_name,
                    "operator_name": op_name,  # backward compat
                    "total_count": op_total,
                    "verified_count": op_verified,
                    "categories": sorted(cats),
                    "sources": sorted(srcs),
                    "frameworks": sorted(fws) if fws else ["pytorch"],
                })

            n_ops = len(operators_out)
            return {
                "layer": layer,
                "total_mr_count": total_mrs,
                "verified_mr_count": verified_mrs,
                "verification_rate": round(verified_mrs / total_mrs, 4) if total_mrs else 0.0,
                "operators_with_mr": n_ops,
                "subjects_with_mr": n_ops,
                "avg_mr_per_operator": round(total_mrs / n_ops, 2) if n_ops else 0.0,
                "avg_mr_per_subject": round(total_mrs / n_ops, 2) if n_ops else 0.0,
                "category_distribution": dict(
                    sorted(category_dist.items(), key=lambda x: -x[1])
                ),
                "source_distribution": dict(
                    sorted(source_dist.items(), key=lambda x: -x[1])
                ),
                "operators": operators_out,
                "subjects": operators_out,
            }
        except Exception as e:
            return {"error": str(e), "operators": [], "subjects": []}

    return _ok(await _acached(cache_key, 60, _load))


@router.get("/mr-repository/{operator_name:path}", summary="单主体 MR 列表")
async def api_mr_detail(operator_name: str, layer_hint: str = "operator"):
    """
    返回指定主体（算子/模型/应用）的全部 MR。
    layer_hint: 层次提示（operator/model/application），默认 operator。
    不缓存（单主体查询成本低）。
    """
    valid_layers = {"operator", "model", "application"}
    layer = layer_hint if layer_hint in valid_layers else "operator"

    def _load():
        mrs = []
        for lyr in [layer] + [l for l in ("operator", "model", "application") if l != layer]:
            repo = MRRepository(layer=lyr)
            mrs = repo.load(operator_name)
            if mrs:
                break
        return [
            {
                "id": mr.id,
                "description": mr.description,
                "transform_code": mr.transform_code,
                "oracle_expr": mr.oracle_expr,
                "category": mr.category,
                "source": mr.source,
                "tolerance": mr.tolerance,
                "layer": mr.layer,
                "lifecycle_state": mr.lifecycle_state,
                "quality_level": mr.quality_level,
                "applicable_frameworks": mr.applicable_frameworks,
                "verified": mr.verified,
                "checked": mr.checked,
                "proven": mr.proven,
                "analysis": mr.analysis,
                "provenance": mr.provenance,
            }
            for mr in mrs
        ]

    try:
        return _ok(await asyncio.to_thread(_load))
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

    return _ok(await _acached("test_results", 10, _load))


@router.get("/test-results/failed", summary="近期失败用例列表")
async def api_test_failed(limit: int = 50):
    """
    读取 test_results 表中的 FAIL 记录，按时间倒序。
    不缓存（便于快速反映最新失败）。
    """
    try:
        rows = await asyncio.to_thread(ResultsManager().get_failed_tests, limit)
        return _ok(rows)
    except Exception:
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

    return _ok(await _acached("evidence", 30, _load))


@router.get("/evidence/{evidence_id}/script", summary="证据包复现脚本")
async def api_evidence_script(evidence_id: str):
    """
    返回指定证据包的完整 Python 复现脚本。
    不缓存（按需访问）。
    """
    def _load():
        pack = EvidenceCollector().load(evidence_id)
        if pack is None:
            raise KeyError(f"证据包 '{evidence_id}' 不存在")
        return {
            "id": pack.evidence_id,
            "operator": pack.operator,
            "mr_description": pack.mr_description,
            "script": pack.reproduce_script,
            "timestamp": pack.timestamp,
        }
    try:
        return _ok(await asyncio.to_thread(_load))
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

    return _ok(await _acached("cross_framework", 60, _load))


@router.get("/cross-framework/{session_id}", summary="跨框架会话详情")
async def api_cross_session(session_id: str):
    """
    返回指定会话的完整数据（含每条 MR 的一致性对比结果）。
    不缓存（按需访问）。
    """
    def _load():
        sessions = CrossFrameworkTester().load_all()
        session = next((s for s in sessions if s.session_id == session_id), None)
        if session is None:
            raise KeyError(f"会话 '{session_id}' 不存在")
        return session.to_dict()
    try:
        return _ok(await asyncio.to_thread(_load))
    except Exception as e:
        return _err(str(e))


# ── K6: 跨层质量视图 ───────────────────────────────────────────────────────────


@router.get("/mr-quality", summary="跨层 MR 质量统计视图")
async def api_mr_quality():
    """
    汇总三层（operator/model/application）MR 仓库的质量分布、来源分布与异常项。
    缓存 60 秒。

    返回字段:
        total_mrs          — 全库 MR 总数
        quality_distribution — 质量等级分布（curated/proven/checked/candidate/retired）
        source_distribution  — 来源分布（llm/template/manual/unknown）
        by_layer           — 各层统计
        anomalies          — 异常告警列表
    """
    def _load():
        try:
            auditor = RepoAuditor()
            report = auditor.run_audit()
            return {
                "total_mrs": report.total_mrs,
                "total_retired": report.total_retired,
                "total_duplicate_groups": report.total_duplicate_groups,
                "quality_distribution": report.quality_distribution(),
                "source_distribution": report.source_distribution(),
                "anomalies": report.anomalies(),
                "by_layer": {
                    lyr: {
                        "total": s.total,
                        "by_quality": s.by_quality,
                        "by_source": s.by_source,
                        "retired": s.retired,
                        "no_oracle": s.no_oracle,
                        "no_provenance": s.no_provenance,
                    }
                    for lyr, s in report.layers.items()
                },
            }
        except Exception as e:
            return {"error": str(e)}

    return _ok(await _acached("mr_quality", 60, _load))


@router.get("/mr-quality/filter", summary="按质量等级筛选 MR")
async def api_mr_quality_filter(
    min_quality: str = "checked",
    layer: Optional[str] = None,
    exclude_retired: bool = True,
):
    """
    按质量等级筛选 MR，返回满足条件的关系列表。

    Query params:
        min_quality     — 最低质量等级（candidate/checked/proven/curated）
        layer           — 可选层次过滤（operator/model/application）
        exclude_retired — 是否排除已退役 MR（默认 True）
    """
    def _load():
        _ql_map = {
            "candidate": "pending", "checked": "checked",
            "proven": "proven", "curated": "curated", "retired": "retired",
        }
        min_ql = QualityLevel.from_lifecycle(_ql_map.get(min_quality, "pending"))
        target_layers = [layer] if layer else ["operator", "model", "application"]

        result = []
        for lyr in target_layers:
            repo = MRRepository(layer=lyr)
            for subject in repo.list_subjects():
                mrs = repo.load(subject)
                filtered = filter_by_quality(
                    mrs, min_quality=min_ql, exclude_retired=exclude_retired
                )
                for m in filtered:
                    result.append({
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
        return {"count": len(result), "mrs": result}

    try:
        return _ok(await asyncio.to_thread(_load))
    except Exception as e:
        return _err(str(e))


# ── Phase P: 三层汇总（summary-v2） ─────────────────────────────────────────────


@router.get("/summary-v2", summary="三层 MR 全量统计摘要")
async def api_summary_v2():
    """
    汇总三层（operator/model/application）MR 仓库的整体规模，
    同时复用 ExperimentOrganizer 的 RQ 数据。缓存 30 秒。
    """
    def _load():
        try:
            layers_stat = {}
            grand_total = 0
            for lyr in ("operator", "model", "application"):
                repo = MRRepository(layer=lyr)
                subjects = repo.list_subjects()
                total = 0
                verified = 0
                for subj in subjects:
                    mrs = repo.load(subj)
                    total += len(mrs)
                    verified += sum(1 for m in mrs if m.verified)
                layers_stat[lyr] = {
                    "total": total,
                    "verified": verified,
                    "subjects": len(subjects),
                    "verification_rate": round(verified / total, 4) if total else 0.0,
                }
                grand_total += total

            rq_data = {}
            try:
                rq_data = ExperimentOrganizer().collect_all()
            except Exception:
                pass

            return {
                "grand_total_mr": grand_total,
                "layers": layers_stat,
                "rq1": rq_data.get("rq1", {}),
                "rq2": rq_data.get("rq2", {}),
                "rq3": rq_data.get("rq3", {}),
                "rq4": rq_data.get("rq4", {}),
            }
        except Exception as e:
            return {"error": str(e), "layers": {}, "grand_total_mr": 0}

    return _ok(await _acached("summary_v2", 30, _load))


# ── Phase P: 框架信息 ──────────────────────────────────────────────────────────


@router.get("/frameworks", summary="已注册框架列表与能力信息")
async def api_frameworks():
    """
    返回各框架的安装状态与版本信息。

    使用 importlib.util.find_spec（文件系统查找）与 importlib.metadata.version
    （读取 .dist-info 元数据）检测框架是否安装及其版本，**不执行任何框架的
    import**，彻底避免 PyTorch / PaddlePaddle 等 C++ 扩展的初始化开销与互斥锁问题。

    缓存 120 秒。
    """
    # 框架静态描述表：name（与 MR applicable_frameworks 对齐）/ pkg（pip 包名）/ optional
    _FW_TABLE = [
        {"name": "pytorch",      "pkg": "torch",      "pip": "torch",        "optional": False},
        {"name": "numpy",        "pkg": "numpy",      "pip": "numpy",        "optional": False},
        {"name": "paddlepaddle", "pkg": "paddle",     "pip": "paddlepaddle", "optional": True},
        {"name": "tensorflow",   "pkg": "tensorflow", "pip": "tensorflow",   "optional": True},
    ]

    def _load():
        import importlib.metadata
        import importlib.util

        # 扫描 MR 仓库统计各框架适用 MR 数（纯文件 I/O，不涉及框架 import）
        fw_mr_counts: Dict[str, int] = {}
        try:
            for lyr in ("operator", "model", "application"):
                repo = MRRepository(layer=lyr)
                for subj in repo.list_subjects():
                    for m in repo.load(subj):
                        for fw in (m.applicable_frameworks or []):
                            fw_mr_counts[fw] = fw_mr_counts.get(fw, 0) + 1
        except Exception:
            pass

        result = []
        for fw in _FW_TABLE:
            # find_spec：只做文件系统查找，不执行任何 Python/C 代码
            available = importlib.util.find_spec(fw["pkg"]) is not None

            version = "N/A"
            if available:
                try:
                    version = importlib.metadata.version(fw["pip"])
                except importlib.metadata.PackageNotFoundError:
                    pass

            result.append({
                "name":               fw["name"],
                "available":          available,
                "optional":           fw["optional"],
                "version":            version,
                "supported_operators": [],   # 枚举算子需 import 框架，仪表盘不需要
                "operator_count":     0,
                "mr_applicable_count": fw_mr_counts.get(fw["name"], 0),
                "status": "available" if available else "unavailable",
            })

        return {"frameworks": result, "total": len(result)}

    return _ok(await _acached("frameworks", 120, _load))


# ── Phase P: 模型层 / 应用层测试结果 ────────────────────────────────────────────


@router.get("/test-results/model", summary="模型层测试结果汇总")
async def api_test_results_model():
    """
    读取 defect_summary 表中 ir_type='model' 的记录。缓存 10 秒。
    """
    def _load():
        try:
            rows = ResultsManager().get_summary()
            model_rows = [r for r in rows if r.get("ir_type") == "model" or
                          r.get("ir_name", "").lower() in
                          ("simplemlp", "simplecnn", "simplernn", "tinytransformer",
                           "resnet18", "vgg16", "mobilenetv2")]
            # 若 ir_type 字段不存在，尝试按模型 registry 名称判断
            if not model_rows:
                try:
                    from deepmt.benchmarks.models.model_registry import ModelBenchmarkRegistry
                    model_names = {m.name.lower() for m in ModelBenchmarkRegistry().list_models()}
                    model_rows = [r for r in rows if r.get("ir_name", "").lower() in model_names]
                except Exception:
                    pass
            return sorted(model_rows, key=lambda r: r.get("failed_tests", 0), reverse=True)
        except Exception:
            return []

    return _ok(await _acached("test_results_model", 10, _load))


@router.get("/test-results/application", summary="应用层验证结果汇总")
async def api_test_results_application():
    """
    读取应用层测试结果（ir_type='application'）。缓存 10 秒。
    """
    def _load():
        try:
            rows = ResultsManager().get_summary()
            app_rows = [r for r in rows if r.get("ir_type") == "application" or
                        r.get("ir_name", "").lower() in
                        ("imageclassification", "textsentiment", "image_classification", "text_sentiment")]
            if not app_rows:
                try:
                    from deepmt.benchmarks.applications.app_registry import ApplicationBenchmarkRegistry
                    app_names = {s.name.lower() for s in ApplicationBenchmarkRegistry().list_scenarios()}
                    app_rows = [r for r in rows if r.get("ir_name", "").lower() in app_names]
                except Exception:
                    pass
            return sorted(app_rows, key=lambda r: r.get("failed_tests", 0), reverse=True)
        except Exception:
            return []

    return _ok(await _acached("test_results_application", 10, _load))


# ── Phase P: 真实缺陷案例 ────────────────────────────────────────────────────────


@router.get("/cases", summary="真实缺陷案例列表")
async def api_cases(status: Optional[str] = None, limit: int = 100):
    """
    列出所有真实缺陷案例（来自 CaseStudyIndex）。
    可按 status 过滤（draft/confirmed/closed）。缓存 30 秒。
    """
    def _load():
        try:
            idx = CaseStudyIndex()
            cases = idx.list_all()
            result = []
            for c in cases[:limit]:
                result.append({
                    "case_id": c.case_id,
                    "operator": c.operator,
                    "framework": c.framework,
                    "framework_version": c.framework_version,
                    "mr_id": c.mr_id,
                    "mr_description": c.mr_description,
                    "layer": c.layer,
                    "defect_type": c.defect_type,
                    "severity": c.severity,
                    "summary": c.summary,
                    "status": c.status,
                    "created_at": c.created_at,
                    "oracle_violation": c.oracle_violation,
                })
            if status:
                result = [r for r in result if r["status"] == status]
            return result
        except Exception as e:
            return []

    return _ok(await _acached("cases", 30, _load))


@router.get("/cases/{case_id}", summary="单案例详情")
async def api_case_detail(case_id: str):
    """返回指定案例的完整信息（不缓存）。"""
    def _load():
        idx = CaseStudyIndex()
        case = idx.load(case_id)
        if case is None:
            raise KeyError(f"案例 '{case_id}' 不存在")
        return case.to_dict()
    try:
        return _ok(await asyncio.to_thread(_load))
    except KeyError as e:
        return _err(str(e))
    except Exception as e:
        return _err(str(e))
