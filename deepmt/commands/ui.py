"""
deepmt ui — Web 仪表盘服务器管理

命令:
    start    启动 Web 仪表盘（默认端口 8080）
"""

import click


@click.group()
def ui():
    """Web 仪表盘服务器管理。"""


@ui.command("start")
@click.option(
    "--port", "-p",
    default=8080,
    show_default=True,
    help="监听端口",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="监听地址（0.0.0.0 表示所有网卡）",
)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    hidden=True,
    help="开发模式：文件变更时自动重载",
)
def ui_start(port: int, host: str, reload: bool) -> None:
    """启动 DeepMT Web 仪表盘。

    \b
    示例:
      deepmt ui start                      # 本地 8080 端口
      deepmt ui start --port 9090          # 自定义端口
      deepmt ui start --host 0.0.0.0       # 局域网可访问

    \b
    前提：已安装 UI 依赖：
      pip install -e ".[ui]"
    """
    try:
        import uvicorn  # noqa: F401
        import fastapi  # noqa: F401
    except ImportError:
        raise click.ClickException(
            "缺少 UI 依赖，请先运行：pip install -e \".[ui]\""
        )

    from deepmt.ui.server import start
    start(host=host, port=port, reload=reload)
