"""日志模块：提供统一的日志记录功能"""

import logging
import os
import sys
import threading
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Literal, Optional

try:
    from wcwidth import wcwidth
except ImportError:

    def wcwidth(char: str) -> int:
        if ord(char[0]) > 0x3000:
            return 2
        return 1


ANSI_RESET = "\033[0m"

CATEGORY_COLORS = {
    "INIT": "\033[38;5;226m",
    "SEARCH": "\033[38;5;39m",
    "LLM": "\033[38;5;213m",
    "OCR": "\033[38;5;215m",
    "GEN": "\033[38;5;51m",
    "CHECK": "\033[38;5;46m",
    "WARN": "\033[38;5;226m",
    "SUCCESS": "\033[38;5;46m",
    "DEBUG": "\033[38;5;245m",
    "INFO": "\033[38;5;39m",
    "WARNING": "\033[38;5;226m",
    "ERROR": "\033[38;5;203m",
    "CRITICAL": "\033[38;5;196m",
    "REPO": "\033[38;5;141m",
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
    "REPO": "📦",
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
    "REPO",
]


class ColoredFormatter(logging.Formatter):
    """终端日志格式化器：添加颜色和图标"""

    def format(self, record):
        # 如果记录包含 category 和 icon，添加颜色格式化
        if hasattr(record, "category") and hasattr(record, "icon"):
            category_upper = record.category.upper()
            icon = record.icon
            color = CATEGORY_COLORS.get(category_upper, "")
            reset = ANSI_RESET

            # 计算图标宽度和填充
            icon_width = sum(wcwidth(c) for c in icon)
            target_width = 7
            category_padding = target_width - icon_width - 1
            if category_padding < 0:
                category_padding = 0

            # 获取原始消息（不修改 record）
            original_msg = record.getMessage()
            # 分离主消息和详细信息（如果有的话）
            lines = original_msg.split("\n")
            main_message = lines[0]
            details = lines[1:] if len(lines) > 1 else []

            # 构建带颜色的 header
            header = f"{color}{icon} {category_upper:<{category_padding}}{reset} | {main_message}"

            # 重新组合消息（不修改 record，直接返回格式化后的字符串）
            if details:
                return header + "\n" + "\n".join(details)
            else:
                return header

        return super().format(record)


class FileFormatter(logging.Formatter):
    """文件日志格式化器：详细格式，纯文本无颜色"""

    def __init__(self):
        super().__init__(
            "%(asctime)s %(levelname)s [%(name)s] %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record):
        # 如果记录包含 category，在消息前添加 [CATEGORY] 标记
        if hasattr(record, "category"):
            # 先格式化基本信息
            formatted = super().format(record)
            # 在消息部分前插入 category（找到 " - " 后的内容）
            parts = formatted.split(" - ", 1)
            if len(parts) == 2:
                return f"{parts[0]} - [{record.category}] {parts[1]}"
            return formatted

        return super().format(record)


class StructuredLogger(logging.Logger):
    """自定义Logger类，自动将标准日志方法转换为结构化日志"""

    def _log_structured(
        self,
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        msg: str,
        args: tuple,
        **kwargs: Any,
    ):
        """内部方法：将标准日志调用转换为 log_structured 调用"""
        # 格式化消息（如果有 args）
        if args:
            msg = msg % args

        # 使用日志级别作为 category
        category: Log_Categories = level  # type: ignore

        # 调用 log_structured（在运行时解析，此时函数已定义）
        log_structured(
            self,
            category,
            msg,
            level=level,
            **kwargs,
        )

    def debug(self, msg, *args, **kwargs):
        """重写 debug 方法"""
        if self.isEnabledFor(logging.DEBUG):
            self._log_structured("DEBUG", msg, args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """重写 info 方法"""
        if self.isEnabledFor(logging.INFO):
            self._log_structured("INFO", msg, args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """重写 warning 方法"""
        if self.isEnabledFor(logging.WARNING):
            self._log_structured("WARNING", msg, args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """重写 error 方法"""
        if self.isEnabledFor(logging.ERROR):
            self._log_structured("ERROR", msg, args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """重写 critical 方法"""
        if self.isEnabledFor(logging.CRITICAL):
            self._log_structured("CRITICAL", msg, args, **kwargs)


class LogManager:
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
        # 设置自定义 Logger 类
        logging.setLoggerClass(StructuredLogger)
        self.log_dir = Path(os.getenv("DEEPMT_LOG_DIR", "data/logs"))
        self.level = getattr(
            logging, os.getenv("DEEPMT_LOG_LEVEL", "INFO").upper(), logging.INFO
        )
        self.console_style = os.getenv("DEEPMT_LOG_CONSOLE_STYLE", "colored").lower()
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

        # 终端 Handler：根据 console_style 选择格式化器
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(self.level)
        if self.console_style == "file":
            console.setFormatter(FileFormatter())
        else:
            console.setFormatter(ColoredFormatter("%(message)s"))

        # 文件 Handler：详细格式，纯文本无颜色
        log_path = self.log_dir / f"deepmt_{datetime.now().strftime('%Y%m%d')}.log"
        file_h = TimedRotatingFileHandler(
            log_path, when="midnight", backupCount=14, encoding="utf-8"
        )
        file_h.setLevel(logging.DEBUG)
        # 使用自定义格式化器，在有 category 时显示它
        file_h.setFormatter(FileFormatter())

        logger.addHandler(console)
        logger.addHandler(file_h)


_manager = LogManager()


def get_logger(name: str = "DeepMT") -> logging.Logger:
    return _manager.get_logger(name)


def reconfigure_logger(
    log_dir: str = "data/logs",
    level: int = logging.INFO,
    console_style: str = "colored",
) -> None:
    _manager.log_dir = Path(log_dir)
    _manager.level = level
    _manager.console_style = console_style.lower()
    for name, logger in _manager._loggers.items():
        _manager._setup_handlers(logger)


def log_structured(
    logger: logging.Logger,
    category: Log_Categories,
    message: str,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    max_detail_width: int = 100,
    **details: Any,
):
    category_upper = category.upper()
    icon = LOG_ICONS.get(category_upper, "•")

    # 构建消息内容（不包含颜色）
    lines = [message]
    if details:
        for key, value in details.items():
            value_str = str(value)
            if len(value_str) > max_detail_width:
                value_str = value_str[: max_detail_width - 3] + "..."
            lines.append(f"  {key:<15} | {value_str}")

    content = "\n".join(lines)

    level_int = getattr(logging, level.upper(), logging.INFO)

    # 通过 extra 参数传递 category 和 icon，让 Formatter 处理颜色
    logger.log(
        level_int,
        content,
        extra={"category": category_upper, "icon": icon},
        stacklevel=2
    )


def log_error(
    logger: logging.Logger,
    message: str,
    exception: Optional[Exception] = None,
    **details: Any,
):
    if exception:
        details["reason"] = str(exception)
        log_structured(
            logger, "ERROR", message, level="ERROR", exc_info=True, **details
        )
    else:
        log_structured(logger, "ERROR", message, level="ERROR", **details)
