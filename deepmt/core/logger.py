"""日志配置：基于 loguru，双 sink（终端 + 文件）。

环境变量：
  DEEPMT_LOG_LEVEL         控制终端输出级别（DEBUG/INFO/WARNING/ERROR），默认 INFO
  DEEPMT_LOG_CONSOLE_STYLE 控制终端格式（colored/file），默认 colored
  DEEPMT_LOG_DIR           日志文件目录，默认 data/logs

DEEPMT_LOG_CONSOLE_STYLE 取值：
  colored — 简洁彩色输出（默认），消息本身携带图标/前缀，不添加时间戳
            示例: 🚀 [INIT] Loaded config from: config.yaml
  file    — 详细纯文本输出，含时间戳/级别/文件/行号（适合日志分析或 CI）
            示例: 2026-01-23 20:00:00 | INFO     | core/config_manager.py:111 — 🚀 [INIT] ...
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger  # noqa: F401  re-export

_level = os.getenv("DEEPMT_LOG_LEVEL", "INFO").upper()
_style = os.getenv("DEEPMT_LOG_CONSOLE_STYLE", "colored").lower()

logger.remove()

if _style == "file":
    # 完整格式：时间戳 + 级别 + 文件:行号 + 消息（无颜色，适合日志分析）
    _console_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {file}:{line} — {message}"
    )
    _colorize = False
elif _level == "DEBUG":
    # DEBUG 模式 colored：加时间戳和文件位置，但保留颜色
    _console_format = (
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<yellow>{file}:{line}</yellow> — {message}"
    )
    _colorize = True
else:
    # 默认 colored：只输出消息本身，消息内的图标/前缀提供上下文
    _console_format = "{message}"
    _colorize = True

logger.add(
    sys.stdout,
    format=_console_format,
    level=_level,
    colorize=_colorize,
)

_log_dir = Path(os.getenv("DEEPMT_LOG_DIR", "data/logs"))
_log_dir.mkdir(parents=True, exist_ok=True)

logger.add(
    _log_dir / f"deepmt_{datetime.now().strftime('%Y%m%d')}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {file}:{line} — {message}",
    level="DEBUG",
    rotation="00:00",
    retention=14,
    encoding="utf-8",
    colorize=False,
)
