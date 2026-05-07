"""
Tests for LiteLLM provider implementation.

Tests cover:
- Provider class attributes and configuration
- Auto-discovery compatibility
- LLM creation via create_llm() with mocked ChatLiteLLM
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


class TestLiteLLMProviderInheritance:
    """Tests for LiteLLMProvider inheritance chain."""

    def test_extends_base_llm_provider(self):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )
        from local_deep_research.llm.providers.base import BaseLLMProvider

        assert issubclass(LiteLLMProvider, BaseLLMProvider)


class TestLiteLLMProviderCreateLLM:
    """Tests for LiteLLMProvider.create_llm() with mocked litellm."""

    @patch(
        "local_deep_research.llm.providers.implementations.litellm.get_setting_from_snapshot"
    )
    @patch(
        "langchain_community.chat_models.litellm.ChatLiteLLM",
    )
    def test_create_llm_returns_chat_litellm(self, mock_chat_cls, mock_settings):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        mock_settings.side_effect = lambda key, default=None, settings_snapshot=None: {
            "llm.litellm.api_key": "sk-test-key",
            "llm.max_tokens": None,
            "llm.streaming": None,
        }.get(key, default)

        mock_instance = MagicMock()
        mock_chat_cls.return_value = mock_instance

        llm = LiteLLMProvider.create_llm(
            model_name="anthropic/claude-sonnet-4-20250514",
            temperature=0.5,
        )

        mock_chat_cls.assert_called_once()
        call_kwargs = mock_chat_cls.call_args.kwargs
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4-20250514"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["api_key"] == "sk-test-key"

    @patch(
        "local_deep_research.llm.providers.implementations.litellm.get_setting_from_snapshot"
    )
    @patch(
        "langchain_community.chat_models.litellm.ChatLiteLLM",
    )
    def test_create_llm_uses_default_model(self, mock_chat_cls, mock_settings):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        mock_settings.return_value = None
        mock_chat_cls.return_value = MagicMock()

        LiteLLMProvider.create_llm()

        call_kwargs = mock_chat_cls.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o"

    @patch(
        "local_deep_research.llm.providers.implementations.litellm.get_setting_from_snapshot"
    )
    @patch(
        "langchain_community.chat_models.litellm.ChatLiteLLM",
    )
    def test_create_llm_omits_api_key_when_not_set(self, mock_chat_cls, mock_settings):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        mock_settings.return_value = None
        mock_chat_cls.return_value = MagicMock()

        LiteLLMProvider.create_llm()

        call_kwargs = mock_chat_cls.call_args.kwargs
        assert "api_key" not in call_kwargs


class TestLiteLLMProviderAvailability:
    """Tests for LiteLLMProvider.is_available()."""

    def test_requires_auth_for_models_returns_false(self):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        assert LiteLLMProvider.requires_auth_for_models() is False

    @patch(
        "local_deep_research.llm.providers.implementations.litellm.get_setting_from_snapshot"
    )
    def test_is_available_with_api_key(self, mock_settings):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        mock_settings.return_value = "sk-test-key"
        assert LiteLLMProvider.is_available() is True

    @patch(
        "local_deep_research.llm.providers.implementations.litellm.get_setting_from_snapshot"
    )
    def test_is_available_with_litellm_installed(self, mock_settings):
        from local_deep_research.llm.providers.implementations.litellm import (
            LiteLLMProvider,
        )

        mock_settings.return_value = None
        # litellm is importable in this test env (or mock it)
        with patch.dict("sys.modules", {"litellm": MagicMock()}):
            assert LiteLLMProvider.is_available() is True


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
