import pytest

from service.config import Settings
from service.providers.embeddings import OpenAIEmbeddings
from service.providers.factory import build_embeddings, build_llm, build_reranker
from service.providers.llm import OpenAILLM
from service.providers.reranker import CohereReranker

pytestmark = pytest.mark.unit


def _settings(monkeypatch, **overrides):
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    monkeypatch.setenv("COHERE_API_KEY", "co")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/ragout")
    return Settings(**overrides)


def test_build_defaults(monkeypatch):
    s = _settings(monkeypatch)
    assert isinstance(build_llm(s), OpenAILLM)
    assert isinstance(build_embeddings(s), OpenAIEmbeddings)
    assert isinstance(build_reranker(s), CohereReranker)


def test_build_unknown_llm_raises(monkeypatch):
    s = _settings(monkeypatch, llm_provider="stub")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_llm(s)


def test_build_unknown_embeddings_raises(monkeypatch):
    s = _settings(monkeypatch, embeddings_provider="stub")
    with pytest.raises(ValueError, match="Unknown embeddings provider"):
        build_embeddings(s)


def test_build_unknown_reranker_raises(monkeypatch):
    s = _settings(monkeypatch, reranker_provider="stub")
    with pytest.raises(ValueError, match="Unknown reranker provider"):
        build_reranker(s)
