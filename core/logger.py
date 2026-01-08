"""日志模块：提供统一的日志记录功能"""

import logging
import re
import sys
import threading
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[38;2;113;119;144m",
        "INFO": "\033[38;2;46;144;188m",
        "WARNING": "\033[38;2;255;165;0m",
        "ERROR": "\033[38;2;220;38;38m",
        "CRITICAL": "\033[38;2;139;0;0m",
    }

    ACCENT = "\033[38;2;88;166;255m"
    MUTED = "\033[38;2;156;163;175m"
    BRIGHT = "\033[38;2;249;250;251m"
    DIM = "\033[38;2;107;114;128m"

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RESET_BOLD = "\033[22m"

    def __init__(
        self, fmt: str, datefmt: Optional[str] = None, enable_color: bool = True
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.enable_color = enable_color

    def format(self, record: logging.LogRecord) -> str:
        if not self.enable_color:
            return super().format(record)

        record = logging.makeLogRecord(record.__dict__)
        levelname = record.levelname
        if levelname in self.COLORS:
            level_prefix = f"{self.COLORS[levelname]}"
            if levelname in ["ERROR", "CRITICAL", "WARNING"]:
                level_prefix += self.BOLD
            record.levelname = f"{level_prefix}{levelname}{self.RESET}"
        record.name = f"{self.MUTED}{record.name}{self.RESET}"
        record.filename = f"{self.ACCENT}{record.filename}{self.RESET}"
        record.module = f"{self.MUTED}{record.module}{self.RESET}"

        result = super().format(record)

        colored_asctime = f"{self.DIM}{record.asctime}{self.RESET}"
        result = result.replace(record.asctime, colored_asctime, 1)
        result = result.replace("DeepMT", f"{self.BRIGHT}DeepMT{self.RESET}")
        return result


class Logger:
    """统一的日志管理器"""

    _default_log_dir: str = "data/logs"
    _default_level: int = logging.INFO

    def __init__(
        self,
        name: str = "DeepMT",
        log_dir: Optional[str] = None,
        level: Optional[int] = None,
    ) -> None:
        """初始化实例属性

        Args:
            name: 日志器名称
            log_dir: 日志文件目录
            level: 日志级别
        """
        self.name = name
        self.log_dir = Path(log_dir or self._default_log_dir)
        self.level = level or self._default_level

        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.propagate = False

        self._apply_config()

    def _apply_config(self):
        """根据配置应用日志设置"""
        if self.logger.handlers:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(
            ColoredFormatter(
                fmt="%(asctime)s %(levelname)s DeepMT %(filename)s:%(lineno)d [%(name)s] %(message)s",
                datefmt="%m-%d %H:%M:%S",
                enable_color=sys.stdout.isatty(),
            )
        )
        log_file = self.log_dir / f"deepmt_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when="midnight",
            interval=1,
            backupCount=14,
            encoding="utf-8",
            utc=False,
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s DeepMT %(filename)s:%(lineno)d [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)


_logger_lock = threading.Lock()
_loggers: dict[str, Logger] = {}


def get_logger(name: str = "DeepMT") -> logging.Logger:
    with _logger_lock:
        if name not in _loggers:
            _loggers[name] = Logger(name=name)
        return _loggers[name].logger


def reconfigure_logger(
    log_dir: str | None = None,
    level: int | None = None,
) -> None:
    with _logger_lock:
        for logger in _loggers.values():
            if log_dir is not None:
                logger.log_dir = Path(log_dir)
            if level is not None:
                logger.level = level

            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)

            logger.logger.setLevel(logger.level)
            logger._apply_config()
