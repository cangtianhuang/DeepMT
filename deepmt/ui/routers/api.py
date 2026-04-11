"""
JSON API 端点

所有端点返回统一结构：
    { "data": ..., "generated_at": "ISO8601", "error": null }

阶段2 实现具体数据端点；阶段1 仅提供健康检查端点。
"""

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter

from deepmt import __version__

router = APIRouter(tags=["API"])


def _ok(data: Any) -> Dict:
    """统一成功响应包装。"""
    return {
        "data": data,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "error": None,
    }


def _err(msg: str) -> Dict:
    """统一错误响应包装。"""
    return {
        "data": None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "error": msg,
    }


# ── 健康检查 ──────────────────────────────────────────────────────────────────


@router.get("/health", summary="服务健康检查")
async def api_health():
    """返回服务状态与版本信息。"""
    return _ok({"status": "ok", "version": __version__})


# ── 数据端点（阶段2实现）─────────────────────────────────────────────────────


@router.get("/summary", summary="RQ1-RQ4 摘要数据")
async def api_summary():
    """RQ1-RQ4 全量摘要（阶段2实现）。"""
    return _ok(None)


@router.get("/mr-repository", summary="MR 仓库列表与统计")
async def api_mr_repository():
    """算子列表 + MR 统计（阶段2实现）。"""
    return _ok(None)


@router.get("/mr-repository/{operator_name:path}", summary="单算子 MR 列表")
async def api_mr_detail(operator_name: str):
    """单算子的 MR 详细列表（阶段2实现）。"""
    return _ok(None)


@router.get("/test-results", summary="测试结果汇总")
async def api_test_results():
    """各算子通过/失败汇总（阶段2实现）。"""
    return _ok(None)


@router.get("/test-results/failed", summary="失败用例列表")
async def api_test_failed():
    """最近失败测试用例（阶段2实现）。"""
    return _ok(None)


@router.get("/evidence", summary="证据包列表")
async def api_evidence():
    """证据包列表（阶段2实现）。"""
    return _ok(None)


@router.get("/evidence/{evidence_id}/script", summary="复现脚本")
async def api_evidence_script(evidence_id: str):
    """单个证据包的 Python 复现脚本（阶段2实现）。"""
    return _ok(None)


@router.get("/cross-framework", summary="跨框架会话列表")
async def api_cross_framework():
    """跨框架一致性会话列表（阶段2实现）。"""
    return _ok(None)


@router.get("/cross-framework/{session_id}", summary="跨框架会话详情")
async def api_cross_session(session_id: str):
    """单次跨框架实验详情（阶段2实现）。"""
    return _ok(None)
