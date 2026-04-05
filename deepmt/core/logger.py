"""日志配置：基于 loguru，双 sink（终端简洁 + 文件详细）。"""

import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger  # noqa: F401  re-export

_level = os.getenv("DEEPMT_LOG_LEVEL", "INFO").upper()
_debug = _level == "DEBUG"

logger.remove()

logger.add(
    sys.stdout,
    format=(
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<yellow>{file}:{line}</yellow> — {message}"
        if _debug else "{message}"
    ),
    level=_level,
    colorize=True,
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
