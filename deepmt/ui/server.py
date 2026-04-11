"""
uvicorn 启动逻辑（供 CLI 调用）

start() 由 deepmt/commands/ui.py 调用，不应直接运行此模块。
"""


def start(host: str = "127.0.0.1", port: int = 8080, reload: bool = False) -> None:
    """
    启动 DeepMT Dashboard 服务器。

    Args:
        host:   监听地址，默认 127.0.0.1（仅本机访问）
        port:   监听端口，默认 8080
        reload: 是否开启热重载（开发模式）
    """
    import uvicorn

    print(f"\n  DeepMT Dashboard 启动中...")
    print(f"  ➜  本地地址：  http://{host}:{port}")
    print(f"  ➜  API 文档：  http://{host}:{port}/api/docs")
    print(f"  ➜  按 Ctrl+C 停止服务器\n")

    uvicorn.run(
        "deepmt.ui.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="warning",   # 仅打印警告，减少演示时的日志噪音
    )
