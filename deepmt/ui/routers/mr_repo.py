"""
MR 知识库页路由

GET /mr          → MR 知识库总览（算子列表 + 分布图）
GET /mr/{operator} → 单算子 MR 详情
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from deepmt.ui.templating import templates

router = APIRouter()


@router.get("/mr", response_class=HTMLResponse)
async def mr_repo(request: Request):
    """MR 知识库页：算子列表、MR 统计、来源/分类分布。"""
    return templates.TemplateResponse(
        request,
        "mr_repo.html",
        context={
            "active_page": "mr_repo",
            "operators": [],
            "stats": {},
        },
    )


@router.get("/mr/{operator_name:path}", response_class=HTMLResponse)
async def mr_detail(request: Request, operator_name: str):
    """单算子 MR 详情页。"""
    return templates.TemplateResponse(
        request,
        "mr_detail.html",
        context={
            "active_page": "mr_repo",
            "operator_name": operator_name,
            "mrs": [],
        },
    )
