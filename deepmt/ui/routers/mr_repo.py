"""
MR 知识库页路由

GET /mr                          → 三层 MR 知识库总览
GET /mr/operator/{name}          → 单算子 MR 详情（旧路径向后兼容）
GET /mr/model/{name}             → 单模型 MR 详情
GET /mr/application/{name}       → 单应用场景 MR 详情
GET /mr/{name:path}              → 旧版路径兜底（算子层）
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from deepmt.ui.templating import templates

router = APIRouter()


@router.get("/mr", response_class=HTMLResponse)
async def mr_repo(request: Request):
    """MR 知识库页：三层主体列表、MR 统计、来源/分类分布。"""
    return templates.TemplateResponse(
        request,
        "mr_repo.html",
        context={"active_page": "mr_repo"},
    )


@router.get("/mr/operator/{subject_name:path}", response_class=HTMLResponse)
async def mr_detail_operator(request: Request, subject_name: str):
    """算子层 MR 详情页。"""
    return templates.TemplateResponse(
        request,
        "mr_detail.html",
        context={
            "active_page": "mr_repo",
            "layer": "operator",
            "subject_name": subject_name,
            "operator_name": subject_name,  # backward compat
        },
    )


@router.get("/mr/model/{subject_name:path}", response_class=HTMLResponse)
async def mr_detail_model(request: Request, subject_name: str):
    """模型层 MR 详情页。"""
    return templates.TemplateResponse(
        request,
        "mr_detail.html",
        context={
            "active_page": "mr_repo",
            "layer": "model",
            "subject_name": subject_name,
            "operator_name": subject_name,
        },
    )


@router.get("/mr/application/{subject_name:path}", response_class=HTMLResponse)
async def mr_detail_application(request: Request, subject_name: str):
    """应用层 MR 详情页。"""
    return templates.TemplateResponse(
        request,
        "mr_detail.html",
        context={
            "active_page": "mr_repo",
            "layer": "application",
            "subject_name": subject_name,
            "operator_name": subject_name,
        },
    )
