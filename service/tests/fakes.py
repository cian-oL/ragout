"""Deterministic fakes for provider Protocols. Used by unit + integration tests."""

from service.providers.base import ScoredText


class FakeLLM:
    """Streams a fixed token sequence; complete() joins them."""

    def __init__(self, tokens: list[str] | None = None) -> None:
        self.tokens = tokens if tokens is not None else ["Hello", " ", "world"]
        self.calls: list = []

    async def stream(self, prompt, history):
        self.calls.append((prompt, history))
        for tok in self.tokens:
            yield tok

    async def complete(self, prompt, history):
        self.calls.append((prompt, history))
        return "".join(self.tokens)


class FakeEmbeddings:
    """Returns unit-axis vectors [1,0,0], [2,0,0], ... per call.

    Deterministic and order-preserving; `start` lets multiple instances
    coexist without colliding indices.
    """

    def __init__(self, start: int = 0, dim: int = 3) -> None:
        self._i = start
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for _ in texts:
            self._i += 1
            out.append([float(self._i)] + [0.0] * (self._dim - 1))
        return out


class FakeReranker:
    """Returns the first `top_n` docs with descending synthetic scores."""

    async def rerank(self, query, docs, top_n):
        scored = [ScoredText(index=i, score=1.0 - i * 0.1) for i in range(len(docs))]
        return scored[:top_n]
