"""
Tests for LiteLLM provider implementation.

Tests cover:
- Provider class attributes and configuration
- Auto-discovery compatibility
- LLM creation via create_llm()
- Availability check
- Model listing
"""

import pytest
from unittest.mock import patch, Mock, MagicMock


class TestLiteLLMProviderAttributes:
    """Tests for LiteLLMProvider class attributes."""

    def test_provider_name(self):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        assert LiteLLMProvider.provider_name == "LiteLLM"

    def test_provider_key(self):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        assert LiteLLMProvider.provider_key == "LITELLM"

    def test_default_base_url(self):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        assert LiteLLMProvider.default_base_url == "http://localhost:4000/v1"

    def test_default_model(self):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        assert LiteLLMProvider.default_model == "openai/gpt-4o"

    def test_is_cloud(self):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        assert LiteLLMProvider.is_cloud is True

    def test_company_name(self):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        assert LiteLLMProvider.company_name == "BerriAI"

    def test_api_key_setting(self):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        assert LiteLLMProvider.api_key_setting == "llm.litellm.api_key"

    def test_url_setting(self):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        assert LiteLLMProvider.url_setting == "llm.litellm.url"


class TestLiteLLMProviderInheritance:
    """Tests for LiteLLMProvider inheritance chain."""

    def test_extends_openai_compatible_provider(self):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )
        from local_deep_research.llm.providers.openai_base import (
            OpenAICompatibleProvider,
        )

        assert issubclass(LiteLLMProvider, OpenAICompatibleProvider)

    def test_extends_base_llm_provider(self):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )
        from local_deep_research.llm.providers.base import BaseLLMProvider

        assert issubclass(LiteLLMProvider, BaseLLMProvider)


class TestLiteLLMProviderCreateLLM:
    """Tests for LiteLLMProvider.create_llm()."""

    @patch(
        "local_deep_research.llm.providers.openai_base.get_setting_from_snapshot"
    )
    def test_create_llm_returns_chat_openai(self, mock_settings):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        mock_settings.side_effect = lambda key, default=None, settings_snapshot=None: {
            "llm.litellm.api_key": "sk-test-key",
            "llm.max_tokens": None,
            "llm.streaming": None,
            "llm.max_retries": None,
            "llm.request_timeout": None,
        }.get(key, default)

        llm = LiteLLMProvider.create_llm(
            model_name="anthropic/claude-sonnet-4-20250514",
            temperature=0.5,
        )

        from langchain_openai import ChatOpenAI

        assert isinstance(llm, ChatOpenAI)

    @patch(
        "local_deep_research.llm.providers.openai_base.get_setting_from_snapshot"
    )
    def test_create_llm_uses_default_model(self, mock_settings):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        mock_settings.side_effect = lambda key, default=None, settings_snapshot=None: {
            "llm.litellm.api_key": "sk-test-key",
            "llm.max_tokens": None,
            "llm.streaming": None,
            "llm.max_retries": None,
            "llm.request_timeout": None,
        }.get(key, default)

        llm = LiteLLMProvider.create_llm()

        assert llm.model_name == "openai/gpt-4o"

    @patch(
        "local_deep_research.llm.providers.openai_base.get_setting_from_snapshot"
    )
    def test_create_llm_uses_default_base_url(self, mock_settings):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        mock_settings.side_effect = lambda key, default=None, settings_snapshot=None: {
            "llm.litellm.api_key": "sk-test-key",
            "llm.max_tokens": None,
            "llm.streaming": None,
            "llm.max_retries": None,
            "llm.request_timeout": None,
        }.get(key, default)

        llm = LiteLLMProvider.create_llm()

        assert "localhost:4000" in str(llm.openai_api_base)


class TestLiteLLMProviderAvailability:
    """Tests for LiteLLMProvider.is_available()."""

    def test_requires_auth_for_models_returns_false(self):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        assert LiteLLMProvider.requires_auth_for_models() is False

    @patch(
        "local_deep_research.llm.providers.openai_base.get_setting_from_snapshot"
    )
    def test_is_available_with_api_key(self, mock_settings):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        mock_settings.return_value = "sk-test-key"

        assert LiteLLMProvider.is_available() is True

    @patch(
        "local_deep_research.llm.providers.openai_base.get_setting_from_snapshot"
    )
    def test_is_not_available_without_api_key(self, mock_settings):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        mock_settings.return_value = None

        assert LiteLLMProvider.is_available() is False


class TestLiteLLMProviderAutoDiscovery:
    """Tests that LiteLLM provider is auto-discovered."""

    def test_provider_in_valid_providers_list(self):
        from local_deep_research.config.llm_config import VALID_PROVIDERS

        assert "litellm" in VALID_PROVIDERS

    def test_provider_has_required_attributes_for_discovery(self):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        assert hasattr(LiteLLMProvider, "provider_name")
        assert hasattr(LiteLLMProvider, "provider_key")
        assert hasattr(LiteLLMProvider, "company_name")
        assert hasattr(LiteLLMProvider, "is_cloud")
        assert LiteLLMProvider.__name__.endswith("Provider")
