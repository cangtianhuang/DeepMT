"""
Jinja2 模板引擎实例（全局共用）

所有路由从此模块导入 `templates`，避免重复实例化。
全局模板变量（version 等）在此统一注入，无需每个路由单独传递。
"""

from pathlib import Path

from fastapi.templating import Jinja2Templates

from deepmt import __version__

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# 全局模板变量：在所有模板中直接使用 {{ version }} 等
templates.env.globals["version"] = __version__
