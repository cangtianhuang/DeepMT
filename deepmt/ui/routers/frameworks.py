"""框架信息页路由  GET /frameworks"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from deepmt.ui.templating import templates

router = APIRouter()


@router.get("/frameworks", response_class=HTMLResponse)
async def frameworks_page(request: Request):
    return templates.TemplateResponse(
        request,
        "frameworks.html",
        context={"active_page": "frameworks"},
    )
