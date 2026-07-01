from unittest.mock import AsyncMock, MagicMock

from service.providers.base import ChatPrompt, EmbeddingsProvider, LLMProvider
from service.providers.embeddings import OpenAIEmbeddings
from service.providers.llm import OpenAILLM


async def test_embeddings_calls_openai(monkeypatch):
    client = MagicMock()
    client.embeddings = MagicMock()
    resp = MagicMock()
    resp.data = [MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])]
    client.embeddings.create = AsyncMock(return_value=resp)
    monkeypatch.setattr(
        "service.providers.embeddings.AsyncOpenAI", lambda *a, **k: client
    )
    prov = OpenAIEmbeddings(api_key="sk", model="text-embedding-3-small")
    out = await prov.embed(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]


async def test_llm_stream(monkeypatch):
    client = MagicMock()

    async def fake_create(**kwargs):
        async def gen():
            m = MagicMock()
            m.choices = [MagicMock(delta=MagicMock(content="hi"))]
            yield m

        return gen()

    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = fake_create
    monkeypatch.setattr("service.providers.llm.AsyncOpenAI", lambda *a, **k: client)
    prov = OpenAILLM(api_key="sk", model="gpt-4o-mini")
    toks = [
        t
        async for t in prov.stream(ChatPrompt(system="s", context=["c"], user="u"), [])
    ]
    assert "".join(toks) == "hi"


async def test_llm_complete(monkeypatch):
    client = MagicMock()

    async def fake_create(**kwargs):
        async def gen():
            m1 = MagicMock()
            m1.choices = [MagicMock(delta=MagicMock(content="he"))]
            yield m1
            m2 = MagicMock()
            m2.choices = [MagicMock(delta=MagicMock(content="llo"))]
            yield m2

        return gen()

    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = fake_create
    monkeypatch.setattr("service.providers.llm.AsyncOpenAI", lambda *a, **k: client)
    prov = OpenAILLM(api_key="sk", model="gpt-4o-mini")
    result = await prov.complete(ChatPrompt(system="s", context=[], user="u"), [])
    assert result == "hello"


async def test_llm_satisfies_protocol():
    prov = OpenAILLM.__new__(OpenAILLM)
    prov._client = MagicMock()
    prov._model = "test"
    assert isinstance(prov, LLMProvider)


async def test_embeddings_satisfies_protocol():
    prov = OpenAIEmbeddings.__new__(OpenAIEmbeddings)
    prov._client = MagicMock()
    prov._model = "test"
    assert isinstance(prov, EmbeddingsProvider)
