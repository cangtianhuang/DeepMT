"""
LLM客户端：通用的LLM调用接口
支持OpenAI、Anthropic等不同提供商
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.logger import get_logger


class LLMClient:
    """
    通用LLM客户端（基于配置的多实例单例模式）

    功能：
    - 统一的LLM调用接口
    - 支持多种LLM提供商
    - 配置管理
    - 错误处理
    - 相同配置返回同一实例，不同配置返回不同实例
    """

    # 类级别的实例字典，key为配置hash，value为实例
    _instances: Dict[str, "LLMClient"] = {}
    # 记录已初始化的配置key
    _initialized_keys: set = set()

    def __new__(
        cls,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """
        创建或获取LLM客户端实例（基于配置的单例）

        Args:
            provider: LLM提供商（openai, anthropic等）
            api_key: API密钥（如果为None，则从配置文件或环境变量获取）
            model: 模型名称（如果为None，则从配置文件获取）
            config_path: 配置文件路径（如果为None则使用默认路径）

        Returns:
            LLMClient实例
        """
        # 生成配置key
        config_key = cls._generate_config_key(provider, api_key, model, config_path)

        # 如果该配置的实例已存在，直接返回
        if config_key in cls._instances:
            return cls._instances[config_key]

        # 创建新实例
        instance = super().__new__(cls)
        cls._instances[config_key] = instance
        return instance

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """
        初始化LLM客户端

        Args:
            provider: LLM提供商（openai, anthropic等）
            api_key: API密钥（如果为None，则从配置文件或环境变量获取）
            model: 模型名称（如果为None，则从配置文件获取）
            config_path: 配置文件路径（如果为None则使用默认路径）
        """
        # 生成配置key
        config_key = self._generate_config_key(provider, api_key, model, config_path)

        # 如果已经初始化过，跳过
        if config_key in LLMClient._initialized_keys:
            return

        self.logger = get_logger()
        self.provider = provider
        self._config_key = config_key

        # 加载配置
        config = self._load_config(config_path)
        llm_config = config.get("llm", {})

        # 获取API key（优先级：参数 > 配置文件 > 环境变量）
        self.api_key = (
            api_key or llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
        )
        self.model = model or llm_config.get("model", "gpt-4")
        self.temperature = llm_config.get("temperature", 0.7)
        self.max_tokens = llm_config.get("max_tokens", 2000)

        if not self.api_key:
            raise ValueError(
                "LLM API key is required! Please:\n"
                "1. Set it in config.yaml (llm.api_key)\n"
                "2. Set OPENAI_API_KEY environment variable\n"
                "3. Pass api_key parameter"
            )

        # 初始化提供商特定的客户端
        self._init_client()

        # 标记为已初始化
        LLMClient._initialized_keys.add(config_key)

    @classmethod
    def _generate_config_key(
        cls,
        provider: str,
        api_key: Optional[str],
        model: Optional[str],
        config_path: Optional[str],
    ) -> str:
        """
        根据配置参数生成唯一的配置key

        Args:
            provider: LLM提供商
            api_key: API密钥
            model: 模型名称
            config_path: 配置文件路径

        Returns:
            配置key字符串
        """
        # 标准化配置参数
        config_str = f"{provider}|{api_key or ''}|{model or ''}|{config_path or ''}"
        # 使用hash生成key
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"

        if not Path(config_path).exists():
            self.logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}")
            return {}

    def _init_client(self):
        """初始化提供商特定的客户端"""
        if self.provider == "openai":
            try:
                import openai

                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai"
                )
        elif self.provider == "anthropic":
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install anthropic"
                )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        发送聊天完成请求

        Args:
            messages: 消息列表，格式：[{"role": "user", "content": "..."}]
            temperature: 温度参数（如果为None则使用配置值）
            max_tokens: 最大token数（如果为None则使用配置值）
            **kwargs: 其他参数

        Returns:
            响应内容字符串
        """
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content.strip()

        elif self.provider == "anthropic":
            # Anthropic API格式不同
            system_msg = None
            user_msgs = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    user_msgs.append(msg["content"])

            response = self.client.messages.create(
                model=self.model,
                system=system_msg,
                messages=[{"role": "user", "content": "\n".join(user_msgs)}],
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **kwargs,
            )
            return response.content[0].text

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
