from unittest.mock import AsyncMock, MagicMock

import pytest

from service.providers.base import Reranker, ScoredText
from service.providers.reranker import CohereReranker

pytestmark = pytest.mark.unit


async def test_rerank_maps_results(monkeypatch):
    client = MagicMock()
    result = MagicMock(index=1, relevance_score=0.9)
    client.rerank = AsyncMock(return_value=MagicMock(results=[result]))
    monkeypatch.setattr(
        "service.providers.reranker.AsyncClient", lambda *a, **k: client
    )
    r = CohereReranker(api_key="co", model="rerank-english-v3.0")
    out = await r.rerank("q", ["a", "b"], top_n=1)
    assert len(out) == 1
    assert out[0].index == 1 and out[0].score == 0.9


async def test_rerank_multiple_results(monkeypatch):
    client = MagicMock()
    results = [
        MagicMock(index=2, relevance_score=0.95),
        MagicMock(index=0, relevance_score=0.6),
    ]
    client.rerank = AsyncMock(return_value=MagicMock(results=results))
    monkeypatch.setattr(
        "service.providers.reranker.AsyncClient", lambda *a, **k: client
    )
    r = CohereReranker(api_key="co", model="rerank-english-v3.0")
    out = await r.rerank("query", ["a", "b", "c"], top_n=2)
    assert len(out) == 2
    assert out[0] == ScoredText(index=2, score=0.95)
    assert out[1] == ScoredText(index=0, score=0.6)


async def test_rerank_passes_args(monkeypatch):
    client = MagicMock()
    client.rerank = AsyncMock(return_value=MagicMock(results=[]))
    monkeypatch.setattr(
        "service.providers.reranker.AsyncClient", lambda *a, **k: client
    )
    r = CohereReranker(api_key="mykey", model="rerank-multilingual-v3.0")
    await r.rerank("test query", ["doc1"], top_n=5)
    client.rerank.assert_awaited_once_with(
        model="rerank-multilingual-v3.0",
        query="test query",
        documents=["doc1"],
        top_n=5,
    )


async def test_reranker_satisfies_protocol():
    r = CohereReranker.__new__(CohereReranker)
    r._client = MagicMock()
    r._model = "test"
    assert isinstance(r, Reranker)
