"""MR 质量视图页路由  GET /quality"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from deepmt.ui.templating import templates

router = APIRouter()


@router.get("/quality", response_class=HTMLResponse)
async def quality_page(request: Request):
    return templates.TemplateResponse(
        request,
        "quality.html",
        context={"active_page": "quality"},
    )
