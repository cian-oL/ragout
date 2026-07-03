"""
OpenAI embeddings provider implementation.
"""

from openai import AsyncOpenAI


class OpenAIEmbeddings:
    """Embeddings provider backed by OpenAI's embeddings API."""

    def __init__(self, api_key: str, model: str):
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors using the configured model."""
        if not texts:
            return []
        response = await self._client.embeddings.create(input=texts, model=self._model)
        return [d.embedding for d in response.data]
