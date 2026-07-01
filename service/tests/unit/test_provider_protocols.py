from service.providers.base import (
    LLMProvider,
    EmbeddingsProvider,
    Reranker,
    ChatPrompt,
    ScoredText,
    ScoredChunk,
)
from fakes import FakeLLM, FakeEmbeddings, FakeReranker


def test_protocols_accept_shared_fakes():
    assert isinstance(FakeLLM(), LLMProvider)
    assert isinstance(FakeEmbeddings(), EmbeddingsProvider)
    assert isinstance(FakeReranker(), Reranker)


def test_chatprompt_fields():
    p = ChatPrompt(system="s", context=["c"], user="u")
    assert p.system == "s" and p.context == ["c"] and p.user == "u"


def test_scoredtext_fields():
    t = ScoredText(index=3, score=0.95)
    assert t.index == 3 and t.score == 0.95


def test_scoredchunk_fields():
    from uuid import UUID

    c = ScoredChunk(
        chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
        document_id=UUID("00000000-0000-0000-0000-000000000002"),
        filename="test.txt",
        ordinal=0,
        text="hello",
        score=0.8,
    )
    assert c.chunk_id == UUID("00000000-0000-0000-0000-000000000001")
    assert c.filename == "test.txt"
    assert c.metadata == {}


async def test_fake_llm_stream():
    llm = FakeLLM(tokens=["a", "b", "c"])
    chunks = [chunk async for chunk in llm.stream(None, [])]
    assert chunks == ["a", "b", "c"]


async def test_fake_llm_complete():
    llm = FakeLLM(tokens=["a", "b", "c"])
    result = await llm.complete(None, [])
    assert result == "abc"


async def test_fake_embeddings():
    emb = FakeEmbeddings(start=0, dim=3)
    result = await emb.embed(["hello", "world"])
    assert result == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]


async def test_fake_reranker():
    reranker = FakeReranker()
    result = await reranker.rerank("query", ["doc1", "doc2", "doc3"], top_n=2)
    assert len(result) == 2
    assert result[0].index == 0
    assert result[0].score > result[1].score
