import pytest

from service.config import Settings


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("COHERE_API_KEY", "co-test")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/ragout")
    s = Settings()
    assert s.llm_provider == "openai"
    assert s.embeddings_provider == "openai"
    assert s.reranker_provider == "cohere"
    assert s.max_upload_bytes == 10485760
    assert s.retrieve_top_k == 20
    assert s.rerank_top_n == 5
    assert s.llm_model == "gpt-4o-mini"
    assert s.embeddings_dim == 1536


def test_settings_requires_openai_key_when_openai_llm(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("COHERE_API_KEY", "co-test")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/ragout")
    with pytest.raises(Exception):
        Settings()


def test_settings_keys_optional_when_providers_swapped(monkeypatch):
    """No API keys required if all providers are 'none' or future-stub values."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/ragout")
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "stub")
    monkeypatch.setenv("RERANKER_PROVIDER", "stub")
    s = Settings()  # should not raise
    assert s.openai_api_key is None
