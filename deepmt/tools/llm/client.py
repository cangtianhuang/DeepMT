"""LLM客户端：通用的LLM调用接口，支持OpenAI、Anthropic等不同提供商"""

import hashlib
import os
import time
from typing import Any, Dict, List, Optional

from deepmt.core.config_manager import get_config_value
from deepmt.core.logger import logger


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
        config_key = self._generate_config_key(provider, api_key, model)
        if config_key in LLMClient._initialized_keys:
            return

        self.provider = provider
        self._config_key = config_key
        self.api_key = (
            api_key or get_config_value("llm.api_key") or os.getenv("OPENAI_API_KEY")
        )
        self.model_base = get_config_value("llm.model_base", "ernie-4.5-turbo-latest")
        self.model_max = get_config_value("llm.model_max", self.model_base)
        self.base_url = get_config_value("llm.url")
        self.temperature = get_config_value("llm.temperature", 0.2)

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
    def _generate_config_key(cls, provider: str, api_key: Optional[str], model: Optional[str]) -> str:
        config_str = f"{provider}|{api_key or ''}|{model or ''}"
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()

    def _init_client(self):
        if self.provider == "openai":
            try:
                import httpx
                import openai
                # 延长 connect timeout（默认 5s 在代理场景下易超时）
                _timeout = httpx.Timeout(timeout=120.0, connect=30.0)
                if self.base_url:
                    self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=_timeout)
                else:
                    self.client = openai.OpenAI(api_key=self.api_key, timeout=_timeout)
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def chat_completion(self, messages: List[Dict[str, Any]], use_model_max: bool = False) -> str:
        model = self.model_max if use_model_max else self.model_base
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        logger.debug(f"🤖 [LLM] Calling {model} | messages={len(messages)} chars={total_chars}")
        logger.info(f"🤖 [LLM] Calling {model} ...")
        start_time = time.time()

        if self.provider == "openai":
            response = self.client.chat.completions.create(  # type: ignore
                model=model, messages=messages, temperature=self.temperature  # type: ignore
            )
            content = response.choices[0].message.content.strip()  # type: ignore
            usage = response.usage
            duration = time.time() - start_time
        elif self.provider == "anthropic":
            system_msg = ""
            user_msgs = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                elif msg["role"] in ("user", "assistant"):
                    user_msgs.append(msg["content"])
            response = self.client.messages.create(  # type: ignore
                model=model,
                system=system_msg,
                messages=[{"role": "user", "content": "\n".join(user_msgs)}],
                temperature=self.temperature,
            )
            content = response.content[0].text
            usage = response.usage
            duration = time.time() - start_time
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        if usage:
            logger.debug(
                f"🤖 [LLM] Response from {model} | "
                f"prompt={usage.prompt_tokens} completion={usage.completion_tokens} "  # type: ignore
                f"total={usage.total_tokens} duration={duration:.1f}s"  # type: ignore
            )
        else:
            logger.debug(f"🤖 [LLM] Response from {model} | duration={duration:.1f}s")
        return content

    def chat_completion_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        use_model_max: bool = False,
    ) -> Dict[str, Any]:
        if self.provider != "openai":
            raise NotImplementedError(
                f"chat_completion_with_tools 目前仅支持 openai 兼容接口，当前 provider: {self.provider}"
            )

        model = self.model_max if use_model_max else self.model_base
        total_chars = sum(len(msg.get("content", "") or "") for msg in messages)
        logger.debug(f"🤖 [LLM] Calling {model} with {len(tools)} tools | chars={total_chars}")
        logger.info(f"🤖 [LLM] Calling {model} with tools ...")
        start_time = time.time()

        response = self.client.chat.completions.create(  # type: ignore
            model=model,
            messages=messages,  # type: ignore
            tools=tools,  # type: ignore
            tool_choice=tool_choice,
            temperature=self.temperature,
        )

        duration = time.time() - start_time
        choice = response.choices[0]
        message = choice.message
        result: Dict[str, Any] = {"content": None, "tool_calls": None}

        if message.tool_calls:
            import json as _json
            parsed_calls = []
            for tc in message.tool_calls:
                try:
                    args = _json.loads(tc.function.arguments)
                except Exception:
                    args = {"_raw": tc.function.arguments}
                parsed_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
            result["tool_calls"] = parsed_calls
            logger.debug(f"🤖 [LLM] Tool calls: {[c['name'] for c in parsed_calls]} | duration={duration:.1f}s")
        else:
            result["content"] = (message.content or "").strip()
            logger.debug(f"🤖 [LLM] Text response | chars={len(result['content'])} duration={duration:.1f}s")

        return result
