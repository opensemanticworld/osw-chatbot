"""Chat model and embedding model used by the chatbot.

Two providers are supported, both served by the same Microsoft Foundry
resource but through different API surfaces:

* ``LLM_PROVIDER=openai`` -> OpenAI-compatible surface
  ``https://<resource>.services.ai.azure.com/openai/v1``. This is what
  ``from openai import OpenAI; OpenAI(base_url=..., api_key=...)`` talks to,
  so no ``api-version`` is involved.
* ``LLM_PROVIDER=claude`` -> Anthropic Messages surface
  ``https://<resource>.services.ai.azure.com/anthropic``. The Anthropic SDK
  appends ``/v1/messages`` and sends the ``x-api-key`` and
  ``anthropic-version`` headers, which is exactly the Foundry contract.

The legacy ``AZURE_OPENAI_ENDPOINT`` + ``AZURE_OPENAI_API_VERSION`` setup is
still honoured when ``OPENAI_BASE_URL`` is not set, so existing ``.env`` files
keep working.
"""

import os

from osw_chatbot.config import LLM_PROVIDER


def _cache():
    """Optional LLM response cache.

    Disabled by default: the cache key is (prompt, model), so with a shared
    cache two different users asking the same question get the exact same
    completion - including one that was generated with the other user's page
    content in context. Set LLM_CACHE_ENABLED=true to opt back in.
    """
    if os.getenv("LLM_CACHE_ENABLED", "false").strip().lower() not in ("1", "true", "yes"):
        return None
    from langchain_community.cache import SQLiteCache

    return SQLiteCache(os.getenv("LLM_CACHE_PATH", "openai_cache.db"))


def _build_openai_llm():
    from langchain_openai import AzureChatOpenAI, ChatOpenAI

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        # Foundry v1 surface - OpenAI shaped, no api_version.
        return ChatOpenAI(
            base_url=base_url,
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.environ["OPENAI_MODEL"],
            cache=_cache(),
        )

    # Legacy Azure OpenAI endpoint (https://<resource>.openai.azure.com).
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        model=os.environ["AZURE_OPENAI_MODEL"],
        cache=_cache(),
    )


def _build_claude_llm():
    from langchain_anthropic import ChatAnthropic

    # Thinking is disabled by default because it breaks multi-step tool
    # calling: the model returns a thinking block with an empty `thinking`
    # field, langchain-anthropic 0.3.17 drops that field when it rebuilds the
    # agent scratchpad, and the next request is rejected with
    # "messages.N.content.0.thinking.thinking: Field required".
    # Set ANTHROPIC_THINKING=adaptive once that is fixed upstream.
    thinking = {"type": os.getenv("ANTHROPIC_THINKING", "disabled").strip()}

    # Note: temperature and top_k are not supported by the Claude models
    # offered in Foundry, and top_p must be >= 0.99 - so none of them are sent.
    return ChatAnthropic(
        base_url=os.environ["ANTHROPIC_BASE_URL"],
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=os.environ["ANTHROPIC_MODEL"],
        max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
        thinking=thinking,
        cache=_cache(),
    )


def build_llm(provider=None):
    provider = (provider or LLM_PROVIDER).lower()
    if provider in ("claude", "anthropic"):
        return _build_claude_llm()
    if provider in ("openai", "azure_openai"):
        return _build_openai_llm()
    raise ValueError(
        f"Unknown LLM_PROVIDER {provider!r}, expected 'openai' or 'claude'"
    )


llm = build_llm()


from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
)
