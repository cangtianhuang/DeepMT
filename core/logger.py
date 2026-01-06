"""日志模块：提供统一的日志记录功能"""

import logging
import sys
from datetime import datetime
from pathlib import Path


class Logger:
    """统一的日志管理器"""

    def __init__(
        self,
        name: str = "DeepMT",
        log_dir: str = "data/logs",
        level: int = logging.INFO,
    ) -> None:
        """初始化实例属性

        Args:
            name: 日志器名称
            log_dir: 日志文件目录
            level: 日志级别
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if self.logger.handlers:
            return

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - DeepMT - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.logger.addHandler(console_handler)

        log_file = self.log_dir / f"deepmt_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - DeepMT - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.logger.addHandler(file_handler)

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
