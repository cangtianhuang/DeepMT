"""
配置加载器：统一的配置文件管理
提供多种配置查找策略，遵循 XDG Base Directory 规范
"""

import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import yaml

from core.logger import get_logger


class ConfigLoader:
    """
    配置加载器：统一的配置管理（单例模式）

    配置文件查找优先级：
    1. 环境变量 DEEPMT_CONFIG_PATH 指定的路径
    2. 当前工作目录的 config.yaml
    3. 项目根目录的 config.yaml
    4. 用户配置目录（~/.config/deepmt/config.yaml）
    """

    _instance: Optional["ConfigLoader"] = None

    # 项目标识文件（用于查找项目根目录）
    PROJECT_MARKERS = (".git", "pyproject.toml", "setup.py", "config.yaml")
    # 向上查找的最大层数
    MAX_PARENT_DEPTH = 10

    def __new__(cls) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_instance()
        return cls._instance

    def _init_instance(self) -> None:
        """初始化实例属性（仅在首次创建时调用）"""
        self.logger = get_logger()
        self._config: Dict[str, Any] = {}
        self._config_path: Optional[Path] = None
        self._config_mtime: float = 0

    def _candidate_paths(self, config_name: str = "config.yaml") -> Iterator[Path]:
        """
        生成候选配置文件路径（按优先级）

        使用生成器模式，避免预先收集所有路径，找到即停止。

        Args:
            config_name: 配置文件名

        Yields:
            候选配置文件路径
        """
        # 策略1: 环境变量指定的路径
        if env_path := os.getenv("DEEPMT_CONFIG_PATH"):
            path = Path(env_path).expanduser().resolve()
            if path.is_file():
                yield path
            elif path.is_dir():
                yield path / config_name

        # 策略2: 当前工作目录
        yield Path.cwd() / config_name

        # 策略3: 项目根目录
        if project_root := self._find_project_root():
            yield project_root / config_name

        # 策略4: 用户配置目录（XDG 规范）
        xdg_config = os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        yield Path(xdg_config) / "deepmt" / config_name

    def _find_project_root(self) -> Optional[Path]:
        """
        向上查找项目根目录（通过项目标识文件）

        Returns:
            项目根目录路径，未找到返回 None
        """
        current = Path(__file__).resolve().parent.parent

        for _ in range(self.MAX_PARENT_DEPTH):
            if any((current / marker).exists() for marker in self.PROJECT_MARKERS):
                return current
            parent = current.parent
            if parent == current:  # 到达文件系统根目录
                break
            current = parent

        return None

    def find_config_file(self, config_name: str = "config.yaml") -> Optional[Path]:
        """
        查找配置文件

        Args:
            config_name: 配置文件名

        Returns:
            配置文件路径，未找到返回 None
        """
        for path in self._candidate_paths(config_name):
            if path.is_file():
                self.logger.debug(f"Found config file: {path}")
                return path

        self.logger.warning(f"Config file '{config_name}' not found")
        return None

    def load(
        self,
        config_path: Optional[Path] = None,
        config_name: str = "config.yaml",
        force_reload: bool = False,
    ) -> Dict[str, Any]:
        """
        加载配置文件

        Args:
            config_path: 明确指定的配置文件路径（可选）
            config_name: 配置文件名（当 config_path 为 None 时使用）
            force_reload: 是否强制重新加载（忽略缓存）

        Returns:
            配置字典，加载失败返回空字典
        """
        # 确定配置文件路径
        if config_path:
            path = Path(config_path).expanduser().resolve()
            if not path.is_file():
                self.logger.warning(f"Config file not found: {path}")
                return {}
        else:
            path = self.find_config_file(config_name)
            if path is None:
                return {}

        # 检查缓存有效性
        if not force_reload and self._is_cache_valid(path):
            return self._config

        # 加载配置
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
            self._config_path = path
            self._config_mtime = path.stat().st_mtime
            self.logger.info(f"Loaded config from: {path}")
            return self._config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}

    def _is_cache_valid(self, path: Path) -> bool:
        """检查配置缓存是否有效"""
        return bool(
            self._config
            and self._config_path == path
            and path.stat().st_mtime == self._config_mtime
        )

    def reload(self) -> Dict[str, Any]:
        """重新加载配置文件"""
        return self.load(self._config_path, force_reload=True)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号路径）

        Args:
            key: 配置键，支持点号分隔路径（如 "llm.api_key"）
            default: 默认值

        Returns:
            配置值，不存在返回默认值

        Examples:
            >>> config.get("llm.api_key")
            >>> config.get("web_search.timeout", 10)
        """
        config = self.load()
        value = config
        for k in key.split("."):
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default

    def section(self, name: str) -> Dict[str, Any]:
        """
        获取配置节

        Args:
            name: 配置节名称（如 "llm", "web_search"）

        Returns:
            配置节字典，不存在返回空字典
        """
        return self.load().get(name, {})

    @property
    def path(self) -> Optional[Path]:
        """当前配置文件路径"""
        return self._config_path


# 全局配置加载器实例
_config_loader = ConfigLoader()


def get_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    获取完整配置

    Args:
        config_path: 配置文件路径（可选）

    Returns:
        配置字典
    """
    return _config_loader.load(config_path)


def get_config_value(key: str, default: Any = None) -> Any:
    """
    获取配置值（支持点号路径）

    Args:
        key: 配置键，支持点号路径（如 "llm.api_key"）
        default: 默认值

    Returns:
        配置值

    Examples:
        >>> get_config_value("llm.api_key")
        >>> get_config_value("web_search.timeout", 10)
    """
    return _config_loader.get(key, default)


def get_config_section(section: str) -> Dict[str, Any]:
    """
    获取配置节

    Args:
        section: 配置节名称

    Returns:
        配置节字典
    """
    return _config_loader.section(section)


def get_config_path() -> Optional[Path]:
    """获取当前配置文件路径"""
    return _config_loader.path


# 常用配置节的便捷访问函数
def get_llm_config() -> Dict[str, Any]:
    """获取 LLM 配置"""
    return get_config_section("llm")


def get_web_search_config() -> Dict[str, Any]:
    """获取网络搜索配置"""
    return get_config_section("web_search")


def get_mr_generation_config() -> Dict[str, Any]:
    """获取 MR 生成配置"""
    return get_config_section("mr_generation")


def get_logging_config() -> Dict[str, Any]:
    """获取日志配置"""
    return get_config_section("logging")
