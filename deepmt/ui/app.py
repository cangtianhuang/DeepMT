"""
DeepMT Web Dashboard — FastAPI 应用实例

构建并返回 FastAPI app，挂载静态资源与所有页面路由。

入口:
    deepmt ui start [--port 8080]

开发模式（热重载）:
    uvicorn deepmt.ui.app:app --reload --port 8080
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from deepmt import __version__
from deepmt.ui.routers import api, cases, cross_framework, frameworks, mr_repo, overview, quality, terminal, test_results

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)


def create_app() -> FastAPI:
    """构建并返回配置完毕的 FastAPI 应用实例。"""
    app = FastAPI(
        title="DeepMT Dashboard",
        description=(
            "深度学习框架蜕变关系自动生成与分层测试体系 — 数据可视化仪表盘。\n\n"
            "提供 RQ1-RQ4 实验数据的只读可视化，包括 MR 知识库浏览、"
            "测试结果面板与跨框架一致性对比。"
        ),
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    # 静态资源（Bootstrap / Chart.js / Bootstrap Icons）
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # 页面路由
    app.include_router(overview.router)
    app.include_router(mr_repo.router)
    app.include_router(test_results.router)
    app.include_router(cross_framework.router)
    app.include_router(frameworks.router)
    app.include_router(quality.router)
    app.include_router(cases.router)
    app.include_router(terminal.router)

    # JSON API（前缀 /api）
    app.include_router(api.router, prefix="/api")

    return app


# 模块级 app 实例供 uvicorn 直接引用
app = create_app()
