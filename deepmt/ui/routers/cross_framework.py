"""
跨框架一致性页路由

GET /cross → 跨框架一致性面板（分组柱图 + 会话列表）
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from deepmt.ui.templating import templates

router = APIRouter()


@router.get("/cross", response_class=HTMLResponse)
async def cross_framework(request: Request):
    """跨框架一致性页：一致率图表 + 会话明细。"""
    return templates.TemplateResponse(
        request,
        "cross_framework.html",
        context={
            "active_page": "cross_framework",
            "sessions": [],
            "stats": {},
        },
    )
