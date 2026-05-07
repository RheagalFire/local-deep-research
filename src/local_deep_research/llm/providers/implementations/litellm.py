"""LiteLLM provider for Local Deep Research.

LiteLLM is an AI gateway that provides access to 100+ LLM providers
(OpenAI, Anthropic, Google, Azure, Bedrock, Ollama, etc.) through a
unified Python SDK — no proxy server needed.

Select "litellm" as the provider in settings and use any model string
supported by LiteLLM (e.g. ``anthropic/claude-sonnet-4-20250514``,
``azure/gpt-4o``, ``bedrock/anthropic.claude-3-haiku``,
``openai/gpt-4o``, ``ollama/llama3``).

See https://docs.litellm.ai/docs/providers for the full model list.

Requires: ``pip install litellm``
"""

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from loguru import logger

from ....config.thread_settings import (
    NoSettingsContextError,
    get_setting_from_snapshot,
)
from ..base import BaseLLMProvider


class _ChatLiteLLM(BaseChatModel):
    """LangChain-compatible chat model backed by the litellm SDK."""

    model: str = "openai/gpt-4o"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    streaming: bool = False

    @property
    def _llm_type(self) -> str:
        return "litellm"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm is required for the LiteLLM provider. "
                "Install with: pip install litellm"
            )

        litellm_messages = _langchain_to_litellm_messages(messages)

        params: dict[str, Any] = {
            "model": self.model,
            "messages": litellm_messages,
            "temperature": self.temperature,
            "drop_params": True,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.api_key:
            params["api_key"] = self.api_key
        if stop:
            params["stop"] = stop
        params.update(kwargs)

        response = litellm.completion(**params)

        content = response.choices[0].message.content or ""
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(
            generations=[generation],
            llm_output={"usage": usage, "model": self.model},
        )


def _langchain_to_litellm_messages(
    messages: List[BaseMessage],
) -> List[dict]:
    """Convert LangChain messages to the OpenAI-style dicts litellm expects."""
    result = []
    for msg in messages:
        if msg.type == "human":
            role = "user"
        elif msg.type == "ai":
            role = "assistant"
        elif msg.type == "system":
            role = "system"
        else:
            role = "user"
        result.append({"role": role, "content": msg.content})
    return result


class LiteLLMProvider(BaseLLMProvider):
    """LiteLLM provider using the litellm SDK directly.

    Routes requests to 100+ LLM providers through LiteLLM's unified
    Python SDK. No proxy server required — just ``pip install litellm``
    and set your provider API keys as environment variables.
    """

    provider_name = "LiteLLM"
    api_key_setting = "llm.litellm.api_key"
    default_model = "openai/gpt-4o"

    # Metadata for auto-discovery
    provider_key = "LITELLM"
    company_name = "BerriAI"
    is_cloud = True

    @classmethod
    def create_llm(cls, model_name=None, temperature=0.7, **kwargs):
        """Create a LangChain-compatible LLM backed by litellm.

        Args:
            model_name: LiteLLM model string
                (e.g. ``anthropic/claude-sonnet-4-20250514``)
            temperature: Model temperature (0.0-1.0)
            **kwargs: Additional arguments including settings_snapshot

        Returns:
            A configured _ChatLiteLLM instance (BaseChatModel)
        """
        settings_snapshot = kwargs.get("settings_snapshot")

        if not model_name:
            model_name = cls.default_model

        litellm_params: dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
        }

        api_key = None
        try:
            api_key = get_setting_from_snapshot(
                cls.api_key_setting,
                default=None,
                settings_snapshot=settings_snapshot,
            )
        except NoSettingsContextError:
            pass
        if api_key:
            litellm_params["api_key"] = api_key

        try:
            max_tokens = get_setting_from_snapshot(
                "llm.max_tokens",
                default=None,
                settings_snapshot=settings_snapshot,
            )
            if max_tokens:
                litellm_params["max_tokens"] = int(max_tokens)
        except NoSettingsContextError:
            pass

        try:
            streaming = get_setting_from_snapshot(
                "llm.streaming",
                default=None,
                settings_snapshot=settings_snapshot,
            )
            if streaming is not None:
                litellm_params["streaming"] = streaming
        except NoSettingsContextError:
            pass

        logger.info(
            f"Creating {cls.provider_name} LLM with model: {model_name}, "
            f"temperature: {temperature}"
        )

        return _ChatLiteLLM(**litellm_params)

    @classmethod
    def is_available(cls, settings_snapshot=None):
        """Check if LiteLLM is available.

        Returns True if either an API key is configured or the litellm
        package is installed (it can pick up provider env vars directly).
        """
        try:
            api_key = get_setting_from_snapshot(
                cls.api_key_setting,
                default=None,
                settings_snapshot=settings_snapshot,
            )
            if api_key and str(api_key).strip():
                return True
        except Exception:
            pass

        try:
            import litellm  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def requires_auth_for_models(cls):
        """LiteLLM doesn't require auth for listing supported models."""
        return False

    @classmethod
    def list_models_for_api(cls, api_key=None, base_url=None):
        """List models supported by LiteLLM."""
        try:
            import litellm

            models = []
            for model_name in sorted(litellm.model_cost.keys()):
                if "/" in model_name:
                    models.append({"value": model_name, "label": model_name})
            return models[:200]
        except ImportError:
            return []
        except Exception:
            logger.debug("Error listing LiteLLM models", exc_info=True)
            return []
