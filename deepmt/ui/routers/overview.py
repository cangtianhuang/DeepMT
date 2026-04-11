"""
总览页路由

GET / → 渲染 overview.html（RQ1-RQ4 KPI 卡片 + 图表）
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from deepmt.ui.templating import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def overview(request: Request):
    """总览页：RQ1-RQ4 指标摘要。"""
    # 阶段2 实现数据加载；Phase 1 传入空上下文骨架
    return templates.TemplateResponse(
        request,
        "overview.html",
        context={
            "active_page": "overview",
            "rq1": {},
            "rq2": {},
            "rq3": {},
            "rq4": {},
        },
    )
