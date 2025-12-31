"""
配置加载器：统一的配置文件管理
提供多种配置查找策略，遵循最佳实践
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from core.logger import get_logger


class ConfigLoader:
    """
    配置加载器：统一的配置管理

    支持多种配置查找策略（按优先级）：
    1. 环境变量 DEEPMT_CONFIG_PATH 指定的路径
    2. 当前工作目录的 config.yaml
    3. 项目根目录的 config.yaml（通过查找项目标识文件）
    4. 用户配置目录（~/.config/deepmt/config.yaml）
    5. 默认配置（如果存在）

    遵循 XDG Base Directory 规范
    """

    _instance: Optional["ConfigLoader"] = None
    _config_cache: Optional[Dict[str, Any]] = None
    _config_path: Optional[Path] = None

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化配置加载器"""
        if hasattr(self, "_initialized"):
            return

        self.logger = get_logger()
        self._initialized = True

    def find_config_file(self, config_name: str = "config.yaml") -> Optional[Path]:
        """
        查找配置文件（按优先级）

        Args:
            config_name: 配置文件名（默认 config.yaml）

        Returns:
            配置文件路径，如果未找到则返回 None
        """
        search_paths = []

        # 策略1: 环境变量指定的路径
        env_path = os.getenv("DEEPMT_CONFIG_PATH")
        if env_path:
            env_path = Path(env_path).expanduser().resolve()
            if env_path.is_file():
                search_paths.append(env_path)
            elif env_path.is_dir():
                search_paths.append(env_path / config_name)

        # 策略2: 当前工作目录
        cwd_config = Path.cwd() / config_name
        if cwd_config.exists():
            search_paths.append(cwd_config)

        # 策略3: 项目根目录（通过查找项目标识文件）
        project_root = self._find_project_root()
        if project_root:
            project_config = project_root / config_name
            if project_config.exists():
                search_paths.append(project_config)

        # 策略4: 用户配置目录（XDG Base Directory 规范）
        xdg_config_home = os.getenv("XDG_CONFIG_HOME")
        if xdg_config_home:
            user_config_dir = Path(xdg_config_home) / "deepmt"
        else:
            user_config_dir = Path.home() / ".config" / "deepmt"

        user_config = user_config_dir / config_name
        if user_config.exists():
            search_paths.append(user_config)

        # 返回第一个找到的配置文件
        for path in search_paths:
            if path.exists() and path.is_file():
                self.logger.debug(f"Found config file: {path}")
                return path

        self.logger.warning(
            f"Config file '{config_name}' not found in any of the search paths"
        )
        return None

    def _find_project_root(self, start_path: Optional[Path] = None) -> Optional[Path]:
        """
        查找项目根目录

        通过查找项目标识文件来确定根目录：
        - .git (Git 仓库)
        - pyproject.toml (Python 项目)
        - setup.py (旧式 Python 项目)
        - README.md (通常位于根目录)

        Args:
            start_path: 起始搜索路径（默认从当前文件位置开始）

        Returns:
            项目根目录路径，如果未找到则返回 None
        """
        if start_path is None:
            # 从当前文件位置开始向上查找
            start_path = Path(__file__).parent.parent

        current = Path(start_path).resolve()

        # 项目标识文件列表
        markers = [".git", "pyproject.toml", "setup.py", "README.md", "config.yaml"]

        # 向上查找，直到找到项目标识文件
        for _ in range(10):  # 最多向上查找10层
            for marker in markers:
                if (current / marker).exists():
                    self.logger.debug(
                        f"Found project root: {current} (marker: {marker})"
                    )
                    return current

            parent = current.parent
            if parent == current:  # 已到达文件系统根目录
                break
            current = parent

        return None

    def load_config(
        self, config_path: Optional[Path] = None, config_name: str = "config.yaml"
    ) -> Dict[str, Any]:
        """
        加载配置文件

        Args:
            config_path: 配置文件路径（如果为None则自动查找）
            config_name: 配置文件名（当config_path为None时使用）

        Returns:
            配置字典，如果加载失败则返回空字典
        """
        # 如果提供了明确的路径，直接使用
        if config_path:
            config_path = Path(config_path).expanduser().resolve()
            if not config_path.exists():
                self.logger.warning(f"Config file not found: {config_path}")
                return {}
        else:
            # 自动查找配置文件
            config_path = self.find_config_file(config_name)
            if config_path is None:
                self.logger.warning(
                    f"Config file '{config_name}' not found, using empty config"
                )
                return {}

        # 检查缓存（如果路径相同）
        if (
            self._config_cache is not None
            and self._config_path == config_path
            and config_path.stat().st_mtime == getattr(self, "_config_mtime", 0)
        ):
            self.logger.debug("Using cached config")
            return self._config_cache

        # 加载配置文件
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            # 更新缓存
            self._config_cache = config
            self._config_path = config_path
            self._config_mtime = config_path.stat().st_mtime

            self.logger.info(f"Loaded config from: {config_path}")
            return config

        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

    def get_config_path(self) -> Optional[Path]:
        """
        获取当前使用的配置文件路径

        Returns:
            配置文件路径，如果未加载则返回 None
        """
        return self._config_path

    def reload_config(self) -> Dict[str, Any]:
        """
        重新加载配置文件（清除缓存）

        Returns:
            配置字典
        """
        if self._config_path:
            self._config_cache = None
            return self.load_config(self._config_path)
        else:
            return self.load_config()

    def get_value(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号路径，如 "llm.api_key"）

        Args:
            key: 配置键，支持点号分隔的路径（如 "llm.api_key"）
            default: 默认值（如果键不存在）

        Returns:
            配置值，如果不存在则返回默认值
        """
        config = self.load_config()
        if not config:
            return default

        # 支持点号路径
        keys = key.split(".")
        value = config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取配置节

        Args:
            section: 配置节名称（如 "llm", "web_search"）

        Returns:
            配置节字典，如果不存在则返回空字典
        """
        config = self.load_config()
        return config.get(section, {})

    def get_llm_config(self) -> Dict[str, Any]:
        """
        获取LLM配置节

        Returns:
            LLM配置字典
        """
        return self.get_section("llm")

    def get_web_search_config(self) -> Dict[str, Any]:
        """
        获取网络搜索配置节

        Returns:
            网络搜索配置字典
        """
        return self.get_section("web_search")

    def get_mr_generation_config(self) -> Dict[str, Any]:
        """
        获取MR生成配置节

        Returns:
            MR生成配置字典
        """
        return self.get_section("mr_generation")

    def get_logging_config(self) -> Dict[str, Any]:
        """
        获取日志配置节

        Returns:
            日志配置字典
        """
        return self.get_section("logging")


# 全局配置加载器实例
_config_loader = ConfigLoader()


def get_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    获取完整配置（便捷函数）

    注意：建议使用 get_config_value() 或 get_*_config() 方法
    来获取特定配置值，而不是获取完整配置字典。

    Args:
        config_path: 配置文件路径（可选）

    Returns:
        配置字典
    """
    return _config_loader.load_config(config_path)


def get_config_value(key: str, default: Any = None) -> Any:
    """
    获取配置值（支持点号路径）

    Args:
        key: 配置键，支持点号分隔的路径（如 "llm.api_key", "web_search.timeout"）
        default: 默认值（如果键不存在）

    Returns:
        配置值，如果不存在则返回默认值

    Examples:
        >>> get_config_value("llm.api_key")
        >>> get_config_value("web_search.timeout", 10)
        >>> get_config_value("web_search.sources.docs", True)
    """
    return _config_loader.get_value(key, default)


def get_config_section(section: str) -> Dict[str, Any]:
    """
    获取配置节

    Args:
        section: 配置节名称（如 "llm", "web_search"）

    Returns:
        配置节字典，如果不存在则返回空字典
    """
    return _config_loader.get_section(section)


def get_llm_config() -> Dict[str, Any]:
    """
    获取LLM配置节

    Returns:
        LLM配置字典
    """
    return _config_loader.get_llm_config()


def get_web_search_config() -> Dict[str, Any]:
    """
    获取网络搜索配置节

    Returns:
        网络搜索配置字典
    """
    return _config_loader.get_web_search_config()


def get_mr_generation_config() -> Dict[str, Any]:
    """
    获取MR生成配置节

    Returns:
        MR生成配置字典
    """
    return _config_loader.get_mr_generation_config()


def get_logging_config() -> Dict[str, Any]:
    """
    获取日志配置节

    Returns:
        日志配置字典
    """
    return _config_loader.get_logging_config()


def get_config_path() -> Optional[Path]:
    """
    获取当前使用的配置文件路径（便捷函数）

    Returns:
        配置文件路径
    """
    return _config_loader.get_config_path()
