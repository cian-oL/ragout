"""
Cohere reranker provider implementation.
"""

from cohere import AsyncClient

from service.providers.base import ScoredText


class CohereReranker:
    """Reranker provider backed by Cohere's rerank API."""

    def __init__(self, api_key: str, model: str):
        self._client = AsyncClient(api_key=api_key)
        self._model = model

    async def rerank(self, query: str, docs: list[str], top_n: int) -> list[ScoredText]:
        """Rerank documents by relevance to query, returning top_n results."""
        resp = await self._client.rerank(
            model=self._model, query=query, documents=docs, top_n=top_n
        )
        return [
            ScoredText(index=r.index, score=r.relevance_score) for r in resp.results
        ]
