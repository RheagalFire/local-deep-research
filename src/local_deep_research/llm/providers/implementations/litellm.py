"""LiteLLM provider for Local Deep Research.

LiteLLM is an AI gateway that provides access to 100+ LLM providers
(OpenAI, Anthropic, Google, Azure, Bedrock, Ollama, etc.) through a
unified OpenAI-compatible API.

Users can either:
1. Run the LiteLLM proxy: ``litellm --model anthropic/claude-sonnet-4-20250514 --port 4000``
2. Or point to an existing LiteLLM proxy deployment

Then select "litellm" as the provider in settings and use any model
string supported by LiteLLM (e.g. ``anthropic/claude-sonnet-4-20250514``,
``azure/gpt-4o``, ``bedrock/anthropic.claude-3-haiku``).

See https://docs.litellm.ai/docs/providers for the full model list.
"""

from ..openai_base import OpenAICompatibleProvider


class LiteLLMProvider(OpenAICompatibleProvider):
    """LiteLLM provider using OpenAI-compatible proxy endpoint.

    Routes requests to 100+ LLM providers through LiteLLM's unified
    API gateway. Supports all models available in LiteLLM including
    OpenAI, Anthropic, Google, Azure, Bedrock, Ollama, and more.
    """

    provider_name = "LiteLLM"
    api_key_setting = "llm.litellm.api_key"
    url_setting = "llm.litellm.url"
    default_base_url = "http://localhost:4000/v1"
    default_model = "openai/gpt-4o"

    # Metadata for auto-discovery
    provider_key = "LITELLM"
    company_name = "BerriAI"
    is_cloud = True

    @classmethod
    def requires_auth_for_models(cls):
        """LiteLLM proxy may not require authentication for listing models."""
        return False
