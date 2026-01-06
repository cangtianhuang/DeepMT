"""日志模块：提供统一的日志记录功能"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


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

        self._is_initialized = False
        self._apply_config()

    def _apply_config(self):
        """根据配置应用日志设置"""
        if self._is_initialized:
            self.logger.handlers.clear()

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger.setLevel(self.level)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - DeepMT - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.logger.addHandler(console_handler)

        log_file = self.log_dir / f"deepmt_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(self.level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - DeepMT - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.logger.addHandler(file_handler)

        self._is_initialized = True

    def reinitialize(self, log_dir: str, level: int):
        self.__class__._default_log_dir = log_dir
        self.__class__._default_level = level
        self.log_dir = Path(log_dir)
        self.level = level
        self._apply_config()

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)


_loggers: dict[str, Logger] = {}


def get_logger(name: str = "DeepMT") -> Logger:
    if name not in _loggers:
        _loggers[name] = Logger(name=name)
    return _loggers[name]
