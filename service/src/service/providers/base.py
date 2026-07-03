"""
Provider protocols, shared dataclasses, and type definitions.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from uuid import UUID


@dataclass
class ChatPrompt:
    """Assembled prompt for LLM calls."""

    system: str
    context: list[str]
    user: str


@dataclass
class ScoredText:
    """A document text with its relevance score from a reranker."""

    index: int
    score: float


@dataclass
class ScoredChunk:
    """A retrieved chunk with its source metadata and relevance score."""

    chunk_id: UUID
    document_id: UUID
    filename: str
    ordinal: int
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers that support streaming and completion."""

    async def stream(self, prompt: ChatPrompt, history: list) -> AsyncIterator[str]:
        """Yield tokens from a streamed LLM response."""
        ...

    async def complete(self, prompt: ChatPrompt, history: list) -> str:
        """Return a complete LLM response as a single string."""
        ...


@runtime_checkable
class EmbeddingsProvider(Protocol):
    """Protocol for text embedding providers."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Protocol for document reranking providers."""

    async def rerank(self, query: str, docs: list[str], top_n: int) -> list[ScoredText]:
        """Rerank documents by relevance to query, returning top_n results."""
        ...
