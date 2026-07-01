from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from uuid import UUID


@dataclass
class ChatPrompt:
    system: str
    context: list[str]
    user: str


@dataclass
class ScoredText:
    index: int
    score: float


@dataclass
class ScoredChunk:
    chunk_id: UUID
    document_id: UUID
    filename: str
    ordinal: int
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    async def stream(self, prompt: ChatPrompt, history: list) -> AsyncIterator[str]: ...
    async def complete(self, prompt: ChatPrompt, history: list) -> str: ...


@runtime_checkable
class EmbeddingsProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self, query: str, docs: list[str], top_n: int
    ) -> list[ScoredText]: ...
