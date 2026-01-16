"""日志模块：提供统一的日志记录功能"""

import logging
import os
import sys
import threading
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Literal, Optional

# --- 样式常量 ---
ANSI_COLORS = {
    "DEBUG": "\033[38;2;148;163;184m",
    "INFO": "\033[38;2;56;189;248m",
    "WARNING": "\033[38;2;250;204;21m",
    "ERROR": "\033[38;2;248;113;113m",
    "CRITICAL": "\033[38;2;220;38;38m",
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
}

LOG_ICONS = {
    "INIT": "🚀",
    "SEARCH": "🔍",
    "LLM": "🤖",
    "OCR": "📷",
    "GEN": "⚡",
    "CHECK": "✅",
    "WARN": "⚠️",
    "SUCCESS": "✨",
    "DEBUG": "🐛",
    "INFO": "💡",
    "WARNING": "🚨",
    "ERROR": "❌",
    "CRITICAL": "❌",
}
Log_Categories = Literal[
    "INIT",
    "SEARCH",
    "LLM",
    "OCR",
    "GEN",
    "CHECK",
    "WARN",
    "SUCCESS",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]


class ModernFormatter(logging.Formatter):

    def __init__(
        self, fmt: str, datefmt: str = "%Y-%m-%d %H:%M:%S", in_terminal: bool = False
    ):
        super().__init__(fmt, datefmt)
        self.in_terminal = in_terminal
        self.project_root = Path(__file__).resolve().parent.parent

    def format(self, record: logging.LogRecord) -> str:
        orig_levelname = record.levelname

        if self.in_terminal:
            color = ANSI_COLORS.get(orig_levelname, "")
            bold = ANSI_COLORS["BOLD"] if record.levelno >= logging.WARNING else ""
            record.levelname = f"{color}{bold}{orig_levelname}{ANSI_COLORS['RESET']}"
            result = record.getMessage()
        else:
            result = super().format(record)

        record.levelname = orig_levelname
        return result


class LogManager:
    """统一的日志管理器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.log_dir = Path(os.getenv("DEEPMT_LOG_DIR", "data/logs"))
        self.level = getattr(
            logging, os.getenv("DEEPMT_LOG_LEVEL", "INFO").upper(), logging.INFO
        )
        self._loggers = {}
        self._initialized = True

    def get_logger(self, name: str) -> logging.Logger:
        with self._lock:
            if name not in self._loggers:
                logger = logging.getLogger(name)
                logger.setLevel(logging.DEBUG)
                logger.propagate = False
                self._setup_handlers(logger)
                self._loggers[name] = logger
            return self._loggers[name]

    def _setup_handlers(self, logger: logging.Logger):
        logger.handlers.clear()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 终端 Handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(self.level)
        console.setFormatter(
            ModernFormatter(fmt="%(message)s", in_terminal=sys.stdout.isatty())
        )

        # 文件 Handler
        log_path = self.log_dir / f"deepmt_{datetime.now().strftime('%Y%m%d')}.log"
        file_h = TimedRotatingFileHandler(
            log_path, when="midnight", backupCount=14, encoding="utf-8"
        )
        file_h.setLevel(logging.DEBUG)
        file_h.setFormatter(
            ModernFormatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] %(filename)s:%(lineno)d - %(message)s",
                in_terminal=False,
            )
        )

        logger.addHandler(console)
        logger.addHandler(file_h)


# --- 工具函数 ---
def get_logger(name: str = "DeepMT") -> logging.Logger:
    return LogManager().get_logger(name)


def reconfigure_logger(log_dir: str = "data/logs", level: int = logging.INFO) -> None:
    manager = LogManager()
    manager.log_dir = Path(log_dir)
    manager.level = level

    for name, logger in manager._loggers.items():
        manager._setup_handlers(logger)


def log_structured(
    logger: logging.Logger,
    category: Log_Categories,
    message: str,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    **details: Any,
):
    icon = LOG_ICONS.get(category.upper(), "•")
    header = f"{icon} {category.upper():<7} | {message}"

    lines = [header]
    for k, v in details.items():
        lines.append(f"  {k:<7} | {v}")

    level_int = getattr(logging, level.upper(), logging.INFO)
    logger.log(level_int, "\n".join(lines), stacklevel=2)


def log_error(
    logger: logging.Logger, message: str, exception: Optional[Exception] = None
):
    if exception:
        logger.error(
            f"{LOG_ICONS['ERROR']} ERROR   | {message} - Reason: {exception}",
            exc_info=True,
            stacklevel=2,
        )
    else:
        logger.error(f"{LOG_ICONS['ERROR']} ERROR   | {message}", stacklevel=2)
