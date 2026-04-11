"""
测试结果页路由

GET /tests → 测试结果面板（通过率图、失败列表、证据包）
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from deepmt.ui.templating import templates

router = APIRouter()


@router.get("/tests", response_class=HTMLResponse)
async def test_results(request: Request):
    """测试结果页：通过率分段柱图 + 失败用例表 + 证据包列表。"""
    return templates.TemplateResponse(
        request,
        "test_results.html",
        context={
            "active_page": "test_results",
            "summary": [],
            "failures": [],
            "evidence": [],
        },
    )
