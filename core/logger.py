"""
日志模块
提供统一的日志记录功能
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    """统一的日志管理器"""
    
    _instance: Optional['Logger'] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, log_dir: str = "data/logs", level: int = logging.INFO):
        """初始化日志系统（单例模式）"""
        if self._initialized:
            return
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建logger
        self.logger = logging.getLogger("DeepMT")
        self.logger.setLevel(level)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 控制台handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)
            
            # 文件handler
            log_file = self.log_dir / f"deepmt_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
        
        self._initialized = True
    
    def debug(self, message: str):
        """记录DEBUG级别日志"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """记录INFO级别日志"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录WARNING级别日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录ERROR级别日志"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """记录CRITICAL级别日志"""
        self.logger.critical(message)


# 全局日志实例
def get_logger() -> Logger:
    """获取全局日志实例"""
    return Logger()


