"""配置加载器：统一的配置文件管理，遵循 XDG Base Directory 规范"""

import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import yaml

from core.logger import get_logger


class ConfigLoader:
    """
    配置加载器：统一的配置管理

    配置文件查找优先级：
    1. 环境变量 DEEPMT_CONFIG_PATH 指定的路径
    2. 当前工作目录的 config.yaml
    3. 项目根目录的 config.yaml
    4. 用户配置目录（~/.config/deepmt/config.yaml）
    """

    _instance: Optional["ConfigLoader"] = None

    PROJECT_MARKERS = (".git", "pyproject.toml", "setup.py", "config.yaml")
    MAX_PARENT_DEPTH = 10

    def __new__(cls) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_instance()
        return cls._instance

    def _init_instance(self) -> None:
        """初始化实例属性"""
        self.logger = get_logger()
        self._config: Dict[str, Any] = self.load()
        self._config_path: Optional[Path] = None
        self._config_mtime: float = 0

    def _candidate_paths(self, config_name: str = "config.yaml") -> Iterator[Path]:
        """生成候选配置文件路径（按优先级）"""
        if env_path := os.getenv("DEEPMT_CONFIG_PATH"):
            path = Path(env_path).expanduser().resolve()
            if path.is_file():
                yield path
            elif path.is_dir():
                yield path / config_name

        yield Path.cwd() / config_name

        if project_root := self._find_project_root():
            yield project_root / config_name

        xdg_config = os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        yield Path(xdg_config) / "deepmt" / config_name

    def _find_project_root(self) -> Optional[Path]:
        """向上查找项目根目录（通过项目标识文件）"""
        current = Path(__file__).resolve().parent.parent

        for _ in range(self.MAX_PARENT_DEPTH):
            if any((current / marker).exists() for marker in self.PROJECT_MARKERS):
                return current
            if (parent := current.parent) == current:
                break
            current = parent

        return None

    def find_config_file(self, config_name: str = "config.yaml") -> Optional[Path]:
        """查找配置文件"""
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
        if config_path:
            path = Path(config_path).expanduser().resolve()
            if not path.is_file():
                self.logger.warning(f"Config file not found: {path}")
                return {}
        else:
            if not (path := self.find_config_file(config_name)):
                return {}

        if not force_reload and self._is_cache_valid(path):
            return self._config

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
        """获取配置值（支持点号路径，如 "llm.api_key"）"""
        value = self._config
        for k in key.split("."):
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default

    def section(self, name: str) -> Dict[str, Any]:
        """获取配置节（如 "llm", "web_search"）"""
        return self._config.get(name, {})

    @property
    def path(self) -> Optional[Path]:
        """当前配置文件路径"""
        return self._config_path


_config_loader = ConfigLoader()


def get_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """获取完整配置"""
    return _config_loader.load(config_path)


def get_config_value(key: str, default: Any = None) -> Any:
    """获取配置值（支持点号路径，如 "llm.api_key"）"""
    return _config_loader.get(key, default)


def get_config_section(section: str) -> Dict[str, Any]:
    """获取配置节"""
    return _config_loader.section(section)


def get_config_path() -> Optional[Path]:
    """获取当前配置文件路径"""
    return _config_loader.path
