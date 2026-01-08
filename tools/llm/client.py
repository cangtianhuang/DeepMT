"""LLM客户端：通用的LLM调用接口，支持OpenAI、Anthropic等不同提供商"""

import hashlib
import os
from typing import Any, Dict, List, Optional

from core.config_loader import get_config_value
from core.logger import get_logger


class LLMClient:
    """通用LLM客户端"""

    _instances: Dict[str, "LLMClient"] = {}
    _initialized_keys: set = set()

    def __new__(
        cls,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """创建或获取LLM客户端实例"""
        config_key = cls._generate_config_key(provider, api_key, model)

        if config_key in cls._instances:
            return cls._instances[config_key]

        instance = super().__new__(cls)
        cls._instances[config_key] = instance
        return instance

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """初始化LLM客户端"""
        config_key = self._generate_config_key(provider, api_key, model)

        if config_key in LLMClient._initialized_keys:
            return

        self.logger = get_logger(self.__class__.__name__)
        self.provider = provider
        self._config_key = config_key
        self.api_key = (
            api_key or get_config_value("llm.api_key") or os.getenv("OPENAI_API_KEY")
        )
        self.model_base = get_config_value("llm.model_base", "ernie-4.5-turbo-latest")
        self.model_max = get_config_value("llm.model_max", "ernie-5.0-thinking-latest")
        self.base_url = get_config_value("llm.url")
        self.temperature = get_config_value("llm.temperature", 0.7)
        self.max_tokens = get_config_value("llm.max_tokens", 2000)

        if not self.api_key:
            raise ValueError(
                "LLM API key is required! Please:\n"
                "1. Set it in config.yaml (llm.api_key)\n"
                "2. Set OPENAI_API_KEY environment variable\n"
                "3. Pass api_key parameter"
            )

        self._init_client()
        LLMClient._initialized_keys.add(config_key)

    @classmethod
    def _generate_config_key(
        cls,
        provider: str,
        api_key: Optional[str],
        model: Optional[str],
    ) -> str:
        """根据配置参数生成唯一的配置key"""
        config_str = f"{provider}|{api_key or ''}|{model or ''}"
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()

    def _init_client(self):
        """初始化提供商特定的客户端"""
        if self.provider == "openai":
            try:
                import openai

                if self.base_url:
                    self.client = openai.OpenAI(
                        api_key=self.api_key, base_url=self.base_url
                    )
                else:
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
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_model_max: bool = False,
        **kwargs,
    ) -> str:
        """
        发送聊天完成请求

        Args:
            messages: 消息列表，格式：[{"role": "user", "content": "..."}]
            temperature: 温度参数（如果为None则使用配置值）
            max_tokens: 最大token数（如果为None则使用配置值）
            use_model_max: 是否使用高级模型（model_max）而非基础模型（model_base）
            **kwargs: 其他参数

        Returns:
            响应内容字符串
        """
        # 根据任务类型选择模型
        model = self.model_max if use_model_max else self.model_base

        messages_preview = "\\n".join(
            [
                f"{msg['role']}: {msg['content'][:100].replace('\n', '\\n')}"
                for msg in messages
            ]
        )
        self.logger.info(f"LLM API called for model {model}: {messages_preview}...")
        if self.provider == "openai":
            response = self.client.chat.completions.create(  # type: ignore
                model=model,
                messages=messages,  # type: ignore
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **kwargs,
            )
            content = response.choices[0].message.content.strip()
            usage = response.usage
            # 将换行符转换为 \n 以便在日志中显示
            content_preview = content[:100].replace("\n", "\\n").replace("\r", "\\r")
            self.logger.info(
                f"LLM API responsed for model {self.model}: {content_preview}..."
                f" (total_tokens={usage.total_tokens})"
            )
            return content

        elif self.provider == "anthropic":
            system_msg = ""
            user_msgs = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                elif msg["role"] in ("user", "assistant"):
                    user_msgs.append(msg["content"])

            response = self.client.messages.create(  # type: ignore
                model=self.model,
                system=system_msg,
                messages=[{"role": "user", "content": "\n".join(user_msgs)}],
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **kwargs,
            )
            content = response.content[0].text
            usage = response.usage
            # 将换行符转换为 \n 以便在日志中显示
            content_preview = content[:100].replace("\n", "\\n").replace("\r", "\\r")
            self.logger.info(
                f"LLM API responsed for model {self.model}: {content_preview}..."
                f" (total_tokens={usage.total_tokens})"
            )
            return content

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
