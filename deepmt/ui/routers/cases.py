"""真实缺陷案例页路由  GET /cases  GET /cases/{case_id}"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from deepmt.ui.templating import templates

router = APIRouter()


@router.get("/cases", response_class=HTMLResponse)
async def cases_page(request: Request):
    return templates.TemplateResponse(
        request,
        "cases.html",
        context={"active_page": "cases"},
    )


@router.get("/cases/{case_id}", response_class=HTMLResponse)
async def case_detail_page(request: Request, case_id: str):
    return templates.TemplateResponse(
        request,
        "cases.html",
        context={"active_page": "cases", "detail_case_id": case_id},
    )
