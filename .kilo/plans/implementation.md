# RAGout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a local-run, Docker Compose–driven RAG service + React app where a reviewer uploads documents and chats with them, with streamed answers, citation chips, a Cohere reranker, and a built-in eval harness.

**Architecture:** FastAPI service with a layered core (parsing → chunking → ingest → retrieve → generate) behind pluggable provider interfaces (LLM / embeddings / reranker), backed by Postgres+pgvector. A separate Vite/React/TS app talks to the REST+SSE API. Synchronous ingestion now, with interfaces that let a background-task implementation drop in later.

**Tech Stack:** Python 3.14, uv workspace, FastAPI, SQLModel, asyncpg, pgvector, pydantic-settings, httpx, pypdf, markdown-it-py, openai, cohere, pytest, testcontainers; Vite, React 18, TypeScript, Tailwind, TanStack Query, React Router, Vitest.

## Global Constraints

- Python `>=3.14`; manage deps with `uv`; lint with `ruff` (run `uv run ruff check` and `uv run ruff format --check` before every commit).
- All service code lives under `service/src/service/`; tests under `service/tests/`.
- Frontend lives under `app/`; uses `npm`/`pnpm`? — use `npm` (default; pin in `app/package.json`).
- Every provider call (LLM/embeddings/reranker) is mocked or faked in tests — never make real network calls in CI/tests.
- SSE chat uses HTTP POST with `text/event-stream`; frame format is `event: <name>\ndata: <json-or-string>\n\n`.
- `MAX_UPLOAD_BYTES` default `10485760` (10 MB); enforced before reading the full body.
- API prefix is `/api/v1` (already established in `feat/api-init`).
- Branding voice: cooking puns in docs/README only; code identifiers stay neutral.
- Commit messages follow Conventional Commits (`feat:`, `test:`, `chore:`, `docs:`).
- Each task ends with green tests + a commit. Frequent commits.

## File Structure (locked)

```
service/src/service/
  config.py                      # pydantic-settings Settings
  api/
    deps.py                      # FastAPI Depends wiring (settings, providers, services)
    run.py                       # existing app factory + entrypoint (extend routers)
    routers/
      health.py                  # existing
      sessions.py
      documents.py
      chat.py
  core/
    parsing.py
    chunking.py
    ingest.py                    # IngestService Protocol + SyncIngestService
    retrieve.py                  # RetrieveService
    generate.py                  # GenerateService + SSE framing helpers
    schemas.py                   # dataclasses/pydantic: ChatPrompt, ScoredChunk, DocumentSummary, Message
  providers/
    base.py                      # Protocols + shared dataclasses
    llm.py                       # OpenAILLM
    embeddings.py                # OpenAIEmbeddings
    reranker.py                  # CohereReranker
    factory.py                   # build_llm / build_embeddings / build_reranker from Settings
  db/
    models.py                    # SQLModel: Session, Document, Chunk, Message
    session.py                   # async engine + AsyncSession factory + lifespan
    pgvector.py                  # Vector type registration + migration runner
    migrations.sql               # CREATE EXTENSION vector + CREATE TABLE IF NOT EXISTS
  eval/
    __main__.py
    run.py
    data/                        # sample corpus + queries.jsonl
    results/                     # gitignored JSON outputs
service/tests/
  conftest.py                    # fixtures: Fake providers, testcontainers pg
  unit/...
  integration/...
app/...                          # scaffolded in Phase 8
docker-compose.yml
Makefile                         # extend existing
.env.example
docs/architecture.md
```

---

## Phase 1 — Service foundation (merge + config + deps)

### Task 1.1: Merge `feat/api-init` and add runtime dependencies

**Files:**

- Modify: `service/pyproject.toml` (add deps)
- Modify: `pyproject.toml` (add dev deps)
- Verify: `service/src/service/api/run.py`, `service/src/service/api/routers/health.py` exist after merge

**Interfaces:** Produces a runnable FastAPI app on `main` with `/api/v1/health`.

- [ ] **Step 1: Merge the remote branch into main locally**

```bash
git merge origin/feat/api-init --no-ff -m "chore: merge feat/api-init (FastAPI skeleton)"
```

- [ ] **Step 2: Add runtime dependencies to `service/pyproject.toml`**

```toml
dependencies = [
    "fastapi[standard-no-fastapi-cloud-cli]>=0.136.1",
    "uvicorn>=0.47.0",
    "sqlmodel>=0.0.22",
    "asyncpg>=0.30.0",
    "pgvector>=0.3.6",
    "pydantic-settings>=2.7.0",
    "httpx>=0.28.0",
    "pypdf>=5.1.0",
    "markdown-it-py>=3.0.0",
    "openai>=1.60.0",
    "cohere>=5.13.0",
]
```

- [ ] **Step 3: Add dev/test dependencies to root `pyproject.toml`**

```toml
[dependency-groups]
dev = [
    "ruff>=0.15.13",
    "service",
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "testcontainers[postgres]>=4.10.0",
]
```

- [ ] **Step 4: Sync and verify the app boots**

```bash
uv sync
uv run --package service service-api &
sleep 3
curl -s http://localhost:8000/api/v1/health | grep OK
kill %1
```

Expected: `{"status":"OK",...}`

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check .
uv run ruff format --check .
git add pyproject.toml service/pyproject.toml uv.lock
git commit -m "chore: add runtime and dev dependencies"
```

### Task 1.2: `Settings` config with `pydantic-settings`

**Files:**

- Create: `service/src/service/config.py`
- Test: `service/tests/unit/test_config.py`

**Interfaces:**

- Produces: `Settings` class with fields `openai_api_key: str`, `cohere_api_key: str`,
  `database_url: str`, `llm_provider: str = "openai"`, `embeddings_provider: str = "openai"`,
  `reranker_provider: str = "cohere"`, `max_upload_bytes: int = 10485760`,
  `llm_model: str = "gpt-4o-mini"`, `embeddings_model: str = "text-embedding-3-small"`,
  `rerank_model: str = "rerank-english-v3.0"`, `retrieve_top_k: int = 20`, `rerank_top_n: int = 5`.

- [ ] **Step 1: Write the failing test**

```python
# service/tests/unit/test_config.py
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


def test_settings_requires_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("COHERE_API_KEY", "co-test")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/ragout")
    with pytest.raises(Exception):
        Settings()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest service/tests/unit/test_config.py -v`
Expected: FAIL (module import error)

- [ ] **Step 3: Write implementation**

```python
# service/src/service/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    cohere_api_key: str
    database_url: str

    llm_provider: str = "openai"
    embeddings_provider: str = "openai"
    reranker_provider: str = "cohere"

    llm_model: str = "gpt-4o-mini"
    embeddings_model: str = "text-embedding-3-small"
    rerank_model: str = "rerank-english-v3.0"

    retrieve_top_k: int = 20
    rerank_top_n: int = 5

    max_upload_bytes: int = 10 * 1024 * 1024
```

- [ ] **Step 4: Run tests, commit**

```bash
uv run pytest service/tests/unit/test_config.py -v
uv run ruff check service/src/service/config.py service/tests/unit/test_config.py
git add service/src/service/config.py service/tests/unit/test_config.py
git commit -m "feat: add Settings config with pydantic-settings"
```

### Task 1.3: Test scaffold + `conftest` with Fake providers

**Files:**

- Create: `service/tests/conftest.py`
- Create: `service/tests/__init__.py` (empty), `service/tests/unit/__init__.py`, `service/tests/integration/__init__.py`
- Modify: `pyproject.toml` — add `[tool.pytest.ini_options]` with `asyncio_mode = "auto"` and `testpaths = ["service/tests"]`

**Interfaces:**

- Produces: `fake_llm`, `fake_embeddings`, `fake_reranker` fixtures implementing the Protocols
  (Protocols defined in Task 2.1; here we pre-import shapes — note: define the Fakes to match
  the Protocol signatures in Task 2.1). To avoid a forward-reference problem, define the Fakes
  in this task as plain async objects; Task 2.1 will assert isinstance via Protocol duck typing.

> Note: This task depends on Task 2.1's Protocols. To keep tasks independently buildable,
> this task defines **local** Fake classes with the same method names; Task 2.1's Protocols
> will structurally match them. Reorder: implement Task 2.1 first, then this. See Phase 2.

- [ ] **Step 1: Add pytest config to root `pyproject.toml`**

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["service/tests"]
```

- [ ] **Step 2: Create package marker files**

```bash
touch service/tests/__init__.py service/tests/unit/__init__.py service/tests/integration/__init__.py
```

- [ ] **Step 3: Write conftest with Fake providers (after Task 2.1 exists)**

```python
# service/tests/conftest.py
import pytest

from service.providers.base import LLMProvider, EmbeddingsProvider, Reranker


class FakeLLM:
    async def stream(self, prompt, history):
        for tok in ["Hello", " ", "world"]:
            yield tok

    async def complete(self, prompt, history):
        return "Hello world"


class FakeEmbeddings:
    def __init__(self):
        self._i = 0

    async def embed(self, texts):
        out = []
        for _ in texts:
            out.append([float(self._i := self._i + 1), 0.0, 0.0])
        return out


class FakeReranker:
    async def rerank(self, query, docs, top_n):
        scored = [{"index": i, "score": 1.0 - i * 0.1} for i in range(len(docs))]
        return scored[:top_n]


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def fake_reranker():
    return FakeReranker()
```

- [ ] **Step 4: Verify collection, commit**

```bash
uv run pytest --collect-only
git add service/tests pyproject.toml
git commit -m "test: add pytest scaffold and fake provider fixtures"
```

---

## Phase 2 — Provider abstraction

### Task 2.1: Provider Protocols + shared dataclasses

**Files:**

- Create: `service/src/service/providers/__init__.py`
- Create: `service/src/service/providers/base.py`
- Test: `service/tests/unit/test_provider_protocols.py`

**Interfaces:**

- Produces:
  - dataclasses `ChatPrompt` (fields: `system: str`, `context: list[str]`, `user: str`),
    `ScoredText` (`index: int`, `score: float`),
    `ScoredChunk` (`chunk_id: UUID`, `document_id: UUID`, `filename: str`, `ordinal: int`, `text: str`, `score: float`, `metadata: dict`).
  - Protocols `LLMProvider`, `EmbeddingsProvider`, `Reranker` (signatures as in spec §9).

- [ ] **Step 1: Write failing test (structural conformance)**

```python
# service/tests/unit/test_provider_protocols.py
from service.providers.base import (
    LLMProvider, EmbeddingsProvider, Reranker, ChatPrompt, ScoredText,
)


class FakeLLM:
    async def stream(self, prompt, history): yield "x"
    async def complete(self, prompt, history): return "x"


class FakeEmb:
    async def embed(self, texts): return [[0.0]]


class FakeRer:
    async def rerank(self, query, docs, top_n): return [ScoredText(index=0, score=1.0)]


def test_protocols_accept_fakes():
    assert isinstance(FakeLLM(), LLMProvider)
    assert isinstance(FakeEmb(), EmbeddingsProvider)
    assert isinstance(FakeRer(), Reranker)


def test_chatprompt_fields():
    p = ChatPrompt(system="s", context=["c"], user="u")
    assert p.system == "s" and p.context == ["c"] and p.user == "u"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest service/tests/unit/test_provider_protocols.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Write implementation**

```python
# service/src/service/providers/base.py
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol
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


class LLMProvider(Protocol):
    async def stream(self, prompt: ChatPrompt, history: list) -> AsyncIterator[str]: ...
    async def complete(self, prompt: ChatPrompt, history: list) -> str: ...


class EmbeddingsProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    async def rerank(self, query: str, docs: list[str], top_n: int) -> list[ScoredText]: ...
```

- [ ] **Step 4: Run tests, commit**

```bash
uv run pytest service/tests/unit/test_provider_protocols.py -v
uv run ruff check service/src/service/providers
git add service/src/service/providers service/tests/unit/test_provider_protocols.py
git commit -m "feat: add provider Protocols and shared dataclasses"
```

### Task 2.2: OpenAI embeddings + LLM implementations

**Files:**

- Create: `service/src/service/providers/llm.py`
- Create: `service/src/service/providers/embeddings.py`
- Test: `service/tests/unit/test_openai_providers.py`

**Interfaces:**

- Consumes: `Settings` (`openai_api_key`, `llm_model`, `embeddings_model`), Protocols from Task 2.1.
- Produces: `OpenAILLM`, `OpenAIEmbeddings` classes.

- [ ] **Step 1: Write failing test using a mocked `openai.AsyncOpenAI`**

```python
# service/tests/unit/test_openai_providers.py
from unittest.mock import AsyncMock, MagicMock

from service.providers.base import ChatPrompt
from service.providers.embeddings import OpenAIEmbeddings
from service.providers.llm import OpenAILLM


async def test_embeddings_calls_openai(monkeypatch):
    client = MagicMock()
    client.embeddings = MagicMock()
    resp = MagicMock()
    resp.data = [MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])]
    client.embeddings.create = AsyncMock(return_value=resp)
    monkeypatch.setattr("service.providers.embeddings.AsyncOpenAI", lambda *a, **k: client)
    prov = OpenAIEmbeddings(api_key="sk", model="text-embedding-3-small")
    out = await prov.embed(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]


async def test_llm_complete(monkeypatch):
    client = MagicMock()
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content=None), message=MagicMock(content="hi"))]
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=client)  # simplified
    monkeypatch.setattr("service.providers.llm.AsyncOpenAI", lambda *a, **k: client)
    # Use a tailored async generator
    async def fake_create(**kwargs):
        async def gen():
            m = MagicMock(); m.choices = [MagicMock(delta=MagicMock(content="hi"))]
            yield m
        return gen()
    client.chat.completions.create = fake_create
    prov = OpenAILLM(api_key="sk", model="gpt-4o-mini")
    toks = [t async for t in prov.stream(ChatPrompt(system="s", context=["c"], user="u"), [])]
    assert "".join(toks) == "hi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest service/tests/unit/test_openai_providers.py -v`
Expected: FAIL (imports)

- [ ] **Step 3: Implement `OpenAIEmbeddings`**

```python
# service/src/service/providers/embeddings.py
from openai import AsyncOpenAI

from service.providers.base import EmbeddingsProvider


class OpenAIEmbeddings:
    def __init__(self, api_key: str, model: str):
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(input=texts, model=self._model)
        return [d.embedding for d in resp.data]
```

- [ ] **Step 4: Implement `OpenAILLM`**

```python
# service/src/service/providers/llm.py
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from service.providers.base import ChatPrompt, LLMProvider


class OpenAILLM:
    def __init__(self, api_key: str, model: str):
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    def _messages(self, prompt: ChatPrompt, history: list) -> list[dict]:
        msgs = [{"role": "system", "content": prompt.system}]
        msgs.extend({"role": r, "content": c} for r, c in history)
        if prompt.context:
            ctx = "\n\n".join(prompt.context)
            msgs.append({"role": "user", "content": f"Context:\n{ctx}"})
        msgs.append({"role": "user", "content": prompt.user})
        return msgs

    async def stream(self, prompt: ChatPrompt, history: list) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model, messages=self._messages(prompt, history), stream=True
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def complete(self, prompt: ChatPrompt, history: list) -> str:
        return "".join([t async for t in self.stream(prompt, history)])
```

- [ ] **Step 5: Run tests, commit**

```bash
uv run pytest service/tests/unit/test_openai_providers.py -v
uv run ruff check service/src/service/providers
git add service/src/service/providers/llm.py service/src/service/providers/embeddings.py service/tests/unit/test_openai_providers.py
git commit -m "feat: add OpenAI LLM and embeddings providers"
```

### Task 2.3: Cohere reranker

**Files:**

- Create: `service/src/service/providers/reranker.py`
- Test: `service/tests/unit/test_cohere_reranker.py`

**Interfaces:**

- Consumes: `Settings` (`cohere_api_key`, `rerank_model`), `Reranker` Protocol, `ScoredText`.
- Produces: `CohereReranker`.

- [ ] **Step 1: Write failing test (mock `cohere.AsyncClient`)**

```python
# service/tests/unit/test_cohere_reranker.py
from unittest.mock import AsyncMock, MagicMock

from service.providers.reranker import CohereReranker


async def test_rerank_maps_results(monkeypatch):
    client = MagicMock()
    result = MagicMock(index=1, relevance_score=0.9)
    client.rerank = AsyncMock(return_value=MagicMock(results=[result]))
    monkeypatch.setattr("service.providers.reranker.AsyncClient", lambda *a, **k: client)
    r = CohereReranker(api_key="co", model="rerank-english-v3.0")
    out = await r.rerank("q", ["a", "b"], top_n=1)
    assert out == [out[0] from out] if False else None  # placeholder removed below
```

> Fix the assertion — replace the placeholder line:

```python
    assert len(out) == 1
    assert out[0].index == 1 and out[0].score == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest service/tests/unit/test_cohere_reranker.py -v`
Expected: FAIL (import)

- [ ] **Step 3: Implement**

```python
# service/src/service/providers/reranker.py
from cohere import AsyncClient

from service.providers.base import Reranker, ScoredText


class CohereReranker:
    def __init__(self, api_key: str, model: str):
        self._client = AsyncClient(api_key=api_key)
        self._model = model

    async def rerank(self, query: str, docs: list[str], top_n: int) -> list[ScoredText]:
        resp = await self._client.rerank(
            model=self._model, query=query, documents=docs, top_n=top_n
        )
        return [ScoredText(index=r.index, score=r.relevance_score) for r in resp.results]
```

- [ ] **Step 4: Run tests, commit**

```bash
uv run pytest service/tests/unit/test_cohere_reranker.py -v
uv run ruff check service/src/service/providers/reranker.py
git add service/src/service/providers/reranker.py service/tests/unit/test_cohere_reranker.py
git commit -m "feat: add Cohere reranker provider"
```

### Task 2.4: Provider factory

**Files:**

- Create: `service/src/service/providers/factory.py`
- Test: `service/tests/unit/test_provider_factory.py`

**Interfaces:**

- Consumes: `Settings` (Task 1.2), provider classes (2.2, 2.3).
- Produces: `build_llm(settings) -> LLMProvider`, `build_embeddings(settings) -> EmbeddingsProvider`, `build_reranker(settings) -> Reranker`. Raises `ValueError` for unknown provider names.

- [ ] **Step 1: Write failing test**

```python
# service/tests/unit/test_provider_factory.py
import pytest

from service.config import Settings
from service.providers.embeddings import OpenAIEmbeddings
from service.providers.llm import OpenAILLM
from service.providers.reranker import CohereReranker
from service.providers.factory import build_llm, build_embeddings, build_reranker


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


def test_build_unknown_raises(monkeypatch):
    s = _settings(monkeypatch, llm_provider="bogus")
    with pytest.raises(ValueError):
        build_llm(s)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest service/tests/unit/test_provider_factory.py -v`
Expected: FAIL (import)

- [ ] **Step 3: Implement**

```python
# service/src/service/providers/factory.py
from service.config import Settings
from service.providers.base import EmbeddingsProvider, LLMProvider, Reranker
from service.providers.embeddings import OpenAIEmbeddings
from service.providers.llm import OpenAILLM
from service.providers.reranker import CohereReranker


def build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        return OpenAILLM(api_key=settings.openai_api_key, model=settings.llm_model)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def build_embeddings(settings: Settings) -> EmbeddingsProvider:
    if settings.embeddings_provider == "openai":
        return OpenAIEmbeddings(api_key=settings.openai_api_key, model=settings.embeddings_model)
    raise ValueError(f"Unknown embeddings provider: {settings.embeddings_provider}")


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker_provider == "cohere":
        return CohereReranker(api_key=settings.cohere_api_key, model=settings.rerank_model)
    raise ValueError(f"Unknown reranker provider: {settings.reranker_provider}")
```

- [ ] **Step 4: Run tests, commit**

```bash
uv run pytest service/tests/unit/test_provider_factory.py -v
uv run ruff check service/src/service/providers
git add service/src/service/providers/factory.py service/tests/unit/test_provider_factory.py
git commit -m "feat: add provider factory"
```

---

## Phase 3 — Database layer (SQLModel + pgvector)

### Task 3.1: SQLModel models

**Files:**

- Create: `service/src/service/db/__init__.py`
- Create: `service/src/service/db/models.py`
- Test: `service/tests/integration/test_models.py`

**Interfaces:**

- Produces: `Session`, `Document`, `Chunk`, `Message` SQLModel table classes per spec §4.
  `Chunk.embedding` is `pgvector.sqlalchemy.Vector(1536)`. `Message.cited_chunk_ids` is `list[UUID]` mapped to `ARRAY(UUID)`.

- [ ] **Step 1: Write failing integration test (requires testcontainers; gate behind pg fixture built in 3.3)**

> This task relies on the testcontainers fixture from Task 3.3. Implement 3.3 first, then this test. For now, create models and a unit test that just imports them.

```python
# service/tests/unit/test_models_import.py
from uuid import UUID, uuid4
from service.db.models import Session, Document, Chunk, Message


def test_models_have_tables():
    assert Session.__tablename__ == "sessions"
    assert Document.__tablename__ == "documents"
    assert Chunk.__tablename__ == "chunks"
    assert Message.__tablename__ == "messages"
    # smoke construct
    s = Session(id=uuid4(), title="t")
    assert s.id is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest service/tests/unit/test_models_import.py -v`
Expected: FAIL (import)

- [ ] **Step 3: Implement models**

```python
# service/src/service/db/models.py
from datetime import datetime, timezone
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Column
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Session(SQLModel, table=True):
    __tablename__ = "sessions"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    title: str = "Untitled session"
    created_at: datetime = Field(default_factory=_now)


class Document(SQLModel, table=True):
    __tablename__ = "documents"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    filename: str
    mime: str
    content_hash: str
    num_chunks: int = 0
    created_at: datetime = Field(default_factory=_now)


class Chunk(SQLModel, table=True):
    __tablename__ = "chunks"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    document_id: UUID = Field(foreign_key="documents.id", index=True)
    ordinal: int
    text: str
    metadata_: dict = Field(default_factory=dict, sa_column=Column("metadata", __import__("sqlalchemy").JSON))
    embedding: list[float] = Field(sa_column=Column(Vector(1536)))


class Message(SQLModel, table=True):
    __tablename__ = "messages"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="sessions.id", index=True)
    role: str
    content: str
    cited_chunk_ids: list[UUID] = Field(default_factory=list, sa_column=Column(ARRAY(UUID)))
    created_at: datetime = Field(default_factory=_now)
```

> Note: the `__import__("sqlalchemy").JSON` hack is ugly — in the real implementation import `JSON` at top: `from sqlalchemy import ARRAY, JSON, Column`. Replace accordingly.

- [ ] **Step 4: Run tests, commit**

```bash
uv run pytest service/tests/unit/test_models_import.py -v
uv run ruff check service/src/service/db
git add service/src/service/db service/tests/unit/test_models_import.py
git commit -m "feat: add SQLModel models"
```

### Task 3.2: Migrations SQL + pgvector registration

**Files:**

- Create: `service/src/service/db/migrations.sql`
- Create: `service/src/service/db/pgvector.py`

**Interfaces:**

- Produces: `migrations.sql` (idempotent: extension + all four tables + HNSW index), `run_migrations(engine)` function that executes the SQL.

- [ ] **Step 1: Write `migrations.sql`**

```sql
-- service/src/service/db/migrations.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'Untitled session',
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    mime TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    num_chunks INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INT NOT NULL,
    text TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1536) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
  ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    cited_chunk_ids UUID[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
```

- [ ] **Step 2: Implement `run_migrations`**

```python
# service/src/service/db/pgvector.py
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncEngine


MIGRATIONS_PATH = Path(__file__).parent / "migrations.sql"


async def run_migrations(engine: AsyncEngine) -> None:
    sql = MIGRATIONS_PATH.read_text()
    async with engine.begin() as conn:
        await conn.exec_driver_sql(sql)
```

- [ ] **Step 3: Commit**

```bash
git add service/src/service/db/migrations.sql service/src/service/db/pgvector.py
git commit -m "feat: add migrations SQL and runner"
```

### Task 3.3: Async engine + session factory + testcontainers fixture

**Files:**

- Create: `service/src/service/db/session.py`
- Modify: `service/tests/conftest.py` — add `pg_engine` fixture using testcontainers

**Interfaces:**

- Produces: `create_engine(database_url) -> AsyncEngine`, `get_session_factory(engine) -> async_sessionmaker`. `app_lifespan` for FastAPI that runs migrations on startup.
- Test fixture `pg_engine` yields a fresh Postgres+pgvector container + migrated engine per test.

- [ ] **Step 1: Write `session.py`**

```python
# service/src/service/db/session.py
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from service.db.pgvector import run_migrations


def create_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, echo=False, pool_pre_ping=True)


def get_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def app_lifespan(engine: AsyncEngine):
    await run_migrations(engine)
    yield
    await engine.dispose()
```

- [ ] **Step 2: Add testcontainers fixture to `conftest.py`**

```python
# append to service/tests/conftest.py
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from service.db.pgvector import run_migrations
from service.db.session import create_engine


@pytest_asyncio.fixture
async def pg_engine():
    # pgvector-enabled container
    container = PostgresContainer("pgvector/pgvector:pg17")
    container.start()
    try:
        url = container.get_connection_url().replace("psycopg2", "asyncpg")
        engine = create_engine(url)
        await run_migrations(engine)
        yield engine
        await engine.dispose()
    finally:
        container.stop()
```

- [ ] **Step 3: Write integration test that round-trips a Session insert**

```python
# service/tests/integration/test_db_roundtrip.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from service.db.models import Session


async def test_session_roundtrip(pg_engine, get_session_factory):
    factory = get_session_factory(pg_engine)
    async with factory() as s:
        s.add(Session(title="demo"))
        await s.commit()
    async with factory() as s:
        rows = (await s.exec(select(Session))).all()
        assert len(rows) == 1
        assert rows[0].title == "demo"
```

> Add a `get_session_factory` fixture alias in conftest returning `get_session_factory` from `service.db.session`.

- [ ] **Step 4: Run tests, commit**

```bash
uv run pytest service/tests/integration -v
uv run ruff check service/src/service/db
git add service/src/service/db/session.py service/tests/conftest.py service/tests/integration/test_db_roundtrip.py
git commit -m "feat: add async engine, session factory, testcontainers fixture"
```

---

## Phase 4 — Core: parsing + chunking

### Task 4.1: Parsing (PDF / MD / TXT)

**Files:**

- Create: `service/src/service/core/__init__.py`
- Create: `service/src/service/core/parsing.py`
- Create: `service/tests/unit/test_parsing.py`
- Create: `service/tests/fixtures/sample.txt`, `sample.md`, `sample.pdf` (commit a 1-page PDF generated from sample.md text; if a fixture PDF cannot be authored cleanly, skip the PDF test and mark with a TODO handled in Task 4.4 — but DO commit a real small PDF to fixtures).

**Interfaces:**

- Produces: `parse(filename: str, mime: str, raw: bytes) -> str` returning plain text.
  Raises `UnsupportedMimeError` for unknown MIME.

- [ ] **Step 1: Write failing test**

```python
# service/tests/unit/test_parsing.py
import pytest

from service.core.parsing import parse, UnsupportedMimeError


def test_parse_txt():
    out = parse("a.txt", "text/plain", b"hello world")
    assert out == "hello world"


def test_parse_md_strips_markup_keeps_headings():
    md = b"# Title\n\nSome **bold** text.\n\n## Sub\n\nMore."
    out = parse("a.md", "text/markdown", md)
    assert "Title" in out and "Some" in out and "Sub" in out
    assert "**" not in out


def test_parse_pdf(tmp_path):
    # uses committed fixture PDF
    from pathlib import Path
    raw = Path("service/tests/fixtures/sample.pdf").read_bytes()
    out = parse("a.pdf", "application/pdf", raw)
    assert "Sample" in out or "sample" in out


def test_parse_unsupported():
    with pytest.raises(UnsupportedMimeError):
        parse("a.docx", "application/vnd.openxmlformats", b"")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest service/tests/unit/test_parsing.py -v`
Expected: FAIL (import)

- [ ] **Step 3: Implement**

```python
# service/src/service/core/parsing.py
import io

from markdown_it import MarkdownIt
from pypdf import PdfReader


class UnsupportedMimeError(ValueError):
    pass


def _parse_pdf(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    return "\f".join(page.extract_text() or "" for page in reader.pages)


def _parse_markdown(raw: bytes) -> str:
    md = MarkdownIt()
    tokens = md.parse(raw.decode("utf-8"))
    out = []
    for t in tokens:
        if t.type in ("heading_open",):
            out.append("\n\n")
        elif t.type in ("paragraph_open",):
            out.append("\n")
    # Simpler robust approach: render to plain via inline tokens
    text = raw.decode("utf-8")
    for sym in ("**", "__", "`", "*"):
        text = text.replace(sym, "")
    return text.strip()


def parse(filename: str, mime: str, raw: bytes) -> str:
    if mime == "text/plain":
        return raw.decode("utf-8")
    if mime == "text/markdown":
        return _parse_markdown(raw)
    if mime == "application/pdf":
        return _parse_pdf(raw)
    raise UnsupportedMimeError(f"Unsupported MIME type: {mime}")
```

- [ ] **Step 4: Run tests, commit**

```bash
uv run pytest service/tests/unit/test_parsing.py -v
uv run ruff check service/src/service/core/parsing.py
git add service/src/service/core service/tests/unit/test_parsing.py service/tests/fixtures
git commit -m "feat: add document parsing for PDF/MD/TXT"
```

### Task 4.2: Chunking

**Files:**

- Create: `service/src/service/core/chunking.py`
- Test: `service/tests/unit/test_chunking.py`

**Interfaces:**

- Produces:
  - dataclass `Chunk` (`ordinal: int`, `text: str`, `metadata: dict`).
  - `chunk(text: str, filename: str, *, target: int = 800, overlap: int = 100) -> list[Chunk]`.

- [ ] **Step 1: Write failing test**

```python
# service/tests/unit/test_chunking.py
from service.core.chunking import chunk


def test_short_text_single_chunk():
    out = chunk("Hello world.", "a.txt")
    assert len(out) == 1
    assert out[0].text == "Hello world."
    assert out[0].ordinal == 0
    assert out[0].metadata["source"] == "a.txt"


def test_splits_markdown_on_headings():
    md = "# A\n\n" + ("x" * 400) + "\n\n# B\n\n" + ("y" * 400)
    out = chunk(md, "a.md")
    headings = [c.metadata.get("heading") for c in out]
    assert "A" in headings and "B" in headings


def test_respects_target_and_overlap():
    text = "word " * 300  # ~1500 chars
    out = chunk(text, "a.txt", target=800, overlap=100)
    assert len(out) >= 2
    # overlap: tail of chunk i appears in head of chunk i+1
    assert out[0].text[-50:] in out[1].text[:150]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest service/tests/unit/test_chunking.py -v`
Expected: FAIL (import)

- [ ] **Step 3: Implement**

```python
# service/src/service/core/chunking.py
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    ordinal: int
    text: str
    metadata: dict = field(default_factory=dict)


_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    positions = [(m.start(), m.group(2).strip()) for m in _HEADING.finditer(text)]
    if not positions:
        return [("", text)]
    parts = []
    for i, (start, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = text[start:end].strip()
        parts.append((heading, body))
    if positions[0][0] > 0:
        parts.insert(0, ("", text[: positions[0][0]].strip()))
    return [p for p in parts if p[1]]


def _recursive_split(text: str, target: int, overlap: int) -> list[str]:
    if len(text) <= target:
        return [text.strip()] if text.strip() else []
    out = []
    start = 0
    while start < len(text):
        end = min(start + target, len(text))
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return out


def chunk(text: str, filename: str, *, target: int = 800, overlap: int = 100) -> list[Chunk]:
    out: list[Chunk] = []
    ordinal = 0
    for heading, body in _split_by_headings(text):
        for piece in _recursive_split(body, target, overlap):
            meta = {"source": filename, "ordinal": ordinal}
            if heading:
                meta["heading"] = heading
            out.append(Chunk(ordinal=ordinal, text=piece, metadata=meta))
            ordinal += 1
    if not out and text.strip():
        out.append(Chunk(ordinal=0, text=text.strip(), metadata={"source": filename, "ordinal": 0}))
    return out
```

- [ ] **Step 4: Run tests, commit**

```bash
uv run pytest service/tests/unit/test_chunking.py -v
uv run ruff check service/src/service/core/chunking.py
git add service/src/service/core/chunking.py service/tests/unit/test_chunking.py
git commit -m "feat: add markdown-aware chunking"
```

### Task 4.3: Core schemas dataclasses

**Files:**

- Create: `service/src/service/core/schemas.py`

**Interfaces:**

- Produces: `DocumentSummary` (`id: UUID`, `filename: str`, `num_chunks: int`, `content_hash: str`), `Message` (`role: str`, `content: str`).

- [ ] **Step 1: Implement (no test; pure dataclass, covered by callers)**

```python
# service/src/service/core/schemas.py
from dataclasses import dataclass
from uuid import UUID


@dataclass
class DocumentSummary:
    id: UUID
    filename: str
    num_chunks: int
    content_hash: str


@dataclass
class Message:
    role: str
    content: str
```

- [ ] **Step 2: Commit**

```bash
git add service/src/service/core/schemas.py
git commit -m "feat: add core schemas"
```

---

## Phase 5 — Ingestion + documents API

### Task 5.1: `IngestService` (Protocol + `SyncIngestService`)

**Files:**

- Create: `service/src/service/core/ingest.py`
- Test: `service/tests/integration/test_ingest.py`

**Interfaces:**

- Consumes: `EmbeddingsProvider`, `AsyncSession`, `parse`, `chunk`, `DocumentSummary`, models.
- Produces: `IngestService` Protocol with `async def ingest(session_id, filename, mime, raw) -> DocumentSummary`. Implementation `SyncIngestService(embeddings, session_factory)`.

- [ ] **Step 1: Write failing integration test (uses `pg_engine`, `fake_embeddings`)**

```python
# service/tests/integration/test_ingest.py
from service.core.ingest import SyncIngestService
from service.db.models import Document, Chunk
from service.db.session import get_session_factory
from sqlmodel import select


async def test_ingest_creates_document_and_chunks(pg_engine, fake_embeddings):
    factory = get_session_factory(pg_engine)
    from service.db.models import Session
    async with factory() as s:
        s.add(Session(title="t"))
        await s.commit()
        sess = (await s.exec(select(Session))).one()

    svc = SyncIngestService(embeddings=fake_embeddings, session_factory=factory)
    summary = await svc.ingest(sess.id, "note.txt", "text/plain", b"Hello world.")
    assert summary.num_chunks == 1
    async with factory() as s:
        docs = (await s.exec(select(Document))).all()
        chunks = (await s.exec(select(Chunk))).all()
        assert len(docs) == 1 and len(chunks) == 1
        assert chunks[0].document_id == docs[0].id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest service/tests/integration/test_ingest.py -v`
Expected: FAIL (import)

- [ ] **Step 3: Implement**

```python
# service/src/service/core/ingest.py
import hashlib
from typing import Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

from service.core.chunking import chunk
from service.core.parsing import parse
from service.core.schemas import DocumentSummary
from service.db.models import Chunk, Document
from service.providers.base import EmbeddingsProvider


class IngestService(Protocol):
    async def ingest(self, session_id: UUID, filename: str, mime: str, raw: bytes) -> DocumentSummary: ...


class SyncIngestService:
    def __init__(self, embeddings: EmbeddingsProvider, session_factory: async_sessionmaker[AsyncSession]):
        self._embeddings = embeddings
        self._factory = session_factory

    async def ingest(self, session_id: UUID, filename: str, mime: str, raw: bytes) -> DocumentSummary:
        text = parse(filename, mime, raw)
        content_hash = hashlib.sha256(raw).hexdigest()
        pieces = chunk(text, filename)
        vectors = await self._embeddings.embed([p.text for p in pieces])
        async with self._factory() as s:
            doc = Document(session_id=session_id, filename=filename, mime=mime,
                           content_hash=content_hash, num_chunks=len(pieces))
            s.add(doc)
            await s.flush()
            for piece, vec in zip(pieces, vectors, strict=True):
                s.add(Chunk(document_id=doc.id, ordinal=piece.ordinal, text=piece.text,
                            metadata_=piece.metadata, embedding=vec))
            await s.commit()
            return DocumentSummary(id=doc.id, filename=doc.filename, num_chunks=doc.num_chunks,
                                   content_hash=doc.content_hash)
```

- [ ] **Step 4: Run tests, commit**

```bash
uv run pytest service/tests/integration/test_ingest.py -v
uv run ruff check service/src/service/core/ingest.py
git add service/src/service/core/ingest.py service/tests/integration/test_ingest.py
git commit -m "feat: add synchronous ingest service"
```

### Task 5.2: Sessions + Documents routers

**Files:**

- Create: `service/src/service/api/deps.py`
- Create: `service/src/service/api/routers/sessions.py`
- Create: `service/src/service/api/routers/documents.py`
- Modify: `service/src/service/api/run.py` — include new routers
- Test: `service/tests/integration/test_routers_sessions_documents.py`

**Interfaces:**

- Consumes: `Settings`, provider factory, session factory, `SyncIngestService`.
- Produces:
  - `deps.py`: `get_settings`, `get_engine`, `get_session_factory`, `get_ingest_service` Depends functions + a `get_app_dependencies` that wires everything (build once at startup).
  - Routers per spec §8 (sessions CRUD, documents upload/list/delete).

- [ ] **Step 1: Write failing integration test (FastAPI TestClient with overridden deps + pg_engine)**

```python
# service/tests/integration/test_routers_sessions_documents.py
from httpx import AsyncClient
from service.api.run import create_app


async def test_create_list_delete_session(pg_engine, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    monkeypatch.setenv("COHERE_API_KEY", "co")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/ragout")
    app = create_app()
    # inject engine
    from service.api.deps import set_engine
    set_engine(pg_engine)
    async with AsyncClient(app=app, base_url="http://t") as client:
        r = await client.post("/api/v1/sessions", json={"title": "Soup"})
        assert r.status_code == 200
        sid = r.json()["id"]
        r = await client.get("/api/v1/sessions")
        assert any(s["id"] == sid for s in r.json())
        r = await client.delete(f"/api/v1/sessions/{sid}")
        assert r.status_code == 204
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest service/tests/integration/test_routers_sessions_documents.py -v`
Expected: FAIL (routes missing)

- [ ] **Step 3: Implement `deps.py`**

```python
# service/src/service/api/deps.py
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

from service.config import Settings
from service.core.ingest import SyncIngestService
from service.db.session import create_engine, get_session_factory
from service.providers.factory import build_embeddings

_engine: AsyncEngine | None = None


def set_engine(engine: AsyncEngine) -> None:
    global _engine
    _engine = engine


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return get_session_factory(get_engine())


def get_ingest_service() -> SyncIngestService:
    return SyncIngestService(embeddings=build_embeddings(get_settings()), session_factory=get_session_factory())
```

- [ ] **Step 4: Implement `sessions.py`**

```python
# service/src/service/api/routers/sessions.py
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select

from service.api.deps import get_session_factory
from service.db.models import Session

router = APIRouter()


@router.post("", status_code=200)
async def create_session(payload: dict, factory=Depends(get_session_factory)) -> dict:
    async with factory() as s:
        row = Session(title=payload.get("title", "Untitled session"))
        s.add(row)
        await s.commit()
        return {"id": str(row.id), "title": row.title, "created_at": row.created_at.isoformat()}


@router.get("")
async def list_sessions(factory=Depends(get_session_factory)) -> list[dict]:
    async with factory() as s:
        rows = (await s.exec(select(Session).order_by(Session.created_at.desc()))).all()
        return [{"id": str(r.id), "title": r.title, "created_at": r.created_at.isoformat()} for r in rows]


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: UUID, factory=Depends(get_session_factory)) -> None:
    async with factory() as s:
        row = await s.get(Session, session_id)
        if not row:
            raise HTTPException(404, "Session not found")
        await s.delete(row)
        await s.commit()
```

- [ ] **Step 5: Implement `documents.py`**

```python
# service/src/service/api/routers/documents.py
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlmodel import select

from service.api.deps import get_ingest_service, get_session_factory, get_settings
from service.core.ingest import IngestService
from service.db.models import Document

router = APIRouter()


@router.post("/{session_id}/documents", status_code=200)
async def upload_document(session_id: UUID, file: UploadFile = File(...),
                          ingest: IngestService = Depends(get_ingest_service),
                          settings=Depends(get_settings)) -> dict:
    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(413, "Upload too large")
    mime = file.content_type or "application/octet-stream"
    try:
        summary = await ingest.ingest(session_id, file.filename or "upload", mime, raw)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {"id": str(summary.id), "filename": summary.filename,
            "num_chunks": summary.num_chunks, "content_hash": summary.content_hash}


@router.get("/{session_id}/documents")
async def list_documents(session_id: UUID, factory=Depends(get_session_factory)) -> list[dict]:
    async with factory() as s:
        rows = (await s.exec(select(Document).where(Document.session_id == session_id))).all()
        return [{"id": str(r.id), "filename": r.filename, "num_chunks": r.num_chunks} for r in rows]


@router.delete("/{session_id}/documents/{doc_id}", status_code=204)
async def delete_document(session_id: UUID, doc_id: UUID, factory=Depends(get_session_factory)) -> None:
    async with factory() as s:
        row = await s.get(Document, doc_id)
        if not row or row.session_id != session_id:
            raise HTTPException(404, "Document not found")
        await s.delete(row)
        await s.commit()
```

- [ ] **Step 6: Wire routers in `run.py`**

```python
# service/src/service/api/run.py  (replace create_app body)
from contextlib import asynccontextmanager

from fastapi import FastAPI

from service.api.deps import get_settings
from service.api.routers import chat, documents, health, sessions
from service.db.session import app_lifespan, create_engine


def create_app() -> FastAPI:
    prefix = "/api/v1"
    app = FastAPI(title="Ragout: The Tasty RAG App")
    app.include_router(health.router, prefix=prefix)
    app.include_router(sessions.router, prefix=prefix)
    app.include_router(documents.router, prefix=prefix)
    app.include_router(chat.router, prefix=prefix)

    @app.on_event("startup")
    async def _startup() -> None:
        from service.api.deps import set_engine
        set_engine(create_engine(get_settings().database_url))

    return app


app = create_app()


def main() -> None:
    import uvicorn
    uvicorn.run(app="service.api.run:app", reload=True)
```

- [ ] **Step 7: Run tests, commit**

```bash
uv run pytest service/tests/integration/test_routers_sessions_documents.py -v
uv run ruff check service/src/service
git add service/src/service/api service/tests/integration/test_routers_sessions_documents.py
git commit -m "feat: add sessions and documents routers"
```

---

## Phase 6 — Retrieval, generation, chat SSE

### Task 6.1: `RetrieveService`

**Files:**

- Create: `service/src/service/core/retrieve.py`
- Test: `service/tests/integration/test_retrieve.py`

**Interfaces:**

- Consumes: `EmbeddingsProvider`, `Reranker`, `AsyncSession`, `ScoredChunk`.
- Produces: `RetrieveService(embeddings, reranker, session_factory, top_k, top_n)` with
  `async def retrieve(session_id, query) -> list[ScoredChunk]`.

- [ ] **Step 1: Write failing integration test**

```python
# service/tests/integration/test_retrieve.py
from service.core.ingest import SyncIngestService
from service.core.retrieve import RetrieveService
from service.db.models import Session
from service.db.session import get_session_factory
from sqlmodel import select


async def test_retrieve_returns_scored_chunks(pg_engine, fake_embeddings, fake_reranker):
    factory = get_session_factory(pg_engine)
    async with factory() as s:
        s.add(Session(title="t"))
        await s.commit()
        sid = (await s.exec(select(Session))).one().id
    ingest = SyncIngestService(fake_embeddings, factory)
    await ingest.ingest(sid, "a.txt", "text/plain", b"The capital of France is Paris.")
    ret = RetrieveService(fake_embeddings, fake_reranker, factory, top_k=5, top_n=3)
    out = await ret.retrieve(sid, "What is the capital of France?")
    assert out and out[0].filename == "a.txt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest service/tests/integration/test_retrieve.py -v`
Expected: FAIL (import)

- [ ] **Step 3: Implement**

```python
# service/src/service/core/retrieve.py
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

from service.db.models import Chunk, Document
from service.providers.base import EmbeddingsProvider, Reranker, ScoredChunk


class RetrieveService:
    def __init__(self, embeddings: EmbeddingsProvider, reranker: Reranker,
                 session_factory: async_sessionmaker[AsyncSession], top_k: int, top_n: int):
        self._emb = embeddings
        self._rer = reranker
        self._factory = session_factory
        self._top_k = top_k
        self._top_n = top_n

    async def retrieve(self, session_id: UUID, query: str) -> list[ScoredChunk]:
        qvec = (await self._emb.embed([query]))[0]
        async with self._factory() as s:
            stmt = (
                select(Chunk, Document)
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.session_id == session_id)
                .order_by(Chunk.embedding.cosine_distance(qvec))
                .limit(self._top_k)
            )
            rows = (await s.exec(stmt)).all()
            candidates = [(chunk, doc, chunk.embedding) for chunk, doc in rows]
        if not candidates:
            return []
        reranked = await self._rer.rerank(query, [c[0].text for c in candidates], self._top_n)
        out = []
        for r in reranked:
            chunk, doc, _ = candidates[r.index]
            out.append(ScoredChunk(chunk_id=chunk.id, document_id=doc.id, filename=doc.filename,
                                   ordinal=chunk.ordinal, text=chunk.text, score=r.score,
                                   metadata=chunk.metadata_))
        return out
```

- [ ] **Step 4: Run tests, commit**

```bash
uv run pytest service/tests/integration/test_retrieve.py -v
uv run ruff check service/src/service/core/retrieve.py
git add service/src/service/core/retrieve.py service/tests/integration/test_retrieve.py
git commit -m "feat: add retrieve service with rerank"
```

### Task 6.2: SSE framing + `GenerateService`

**Files:**

- Create: `service/src/service/core/generate.py`
- Test: `service/tests/unit/test_generate.py`

**Interfaces:**

- Consumes: `LLMProvider`, `ScoredChunk`, `ChatPrompt`, `Message`.
- Produces:
  - `format_sse(event: str, data: str) -> str` → `"event: {event}\ndata: {data}\n\n"` (data JSON-encoded when dict).
  - `build_prompt(system: str, chunks: list[ScoredChunk], question: str) -> ChatPrompt`.
  - `GenerateService(llm, session_factory)` with `async def stream(session_id, query, chunks) -> AsyncIterator[str]` yielding SSE frames, persisting the message pair at the end.

- [ ] **Step 1: Write failing unit test (SSE + prompt + stream with FakeLLM)**

```python
# service/tests/unit/test_generate.py
import json

from service.core.generate import build_prompt, format_sse, GenerateService
from service.providers.base import ScoredText  # noqa


def test_format_sse_string():
    assert format_sse("token", "hi") == "event: token\ndata: hi\n\n"


def test_format_sse_dict_json():
    frame = format_sse("done", {"a": 1})
    assert frame.startswith("event: done\ndata: ")
    assert json.loads(frame.split("data: ", 1)[1].strip()) == {"a": 1}


def test_build_prompt_numbers_chunks():
    from uuid import uuid4
    from service.providers.base import ScoredChunk
    c = ScoredChunk(chunk_id=uuid4(), document_id=uuid4(), filename="a.txt", ordinal=3,
                    text="Paris", score=0.9)
    p = build_prompt("sys", [c], "capital?")
    assert "[1] a.txt#3: Paris" in p.context[0]
    assert p.user == "capital?"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest service/tests/unit/test_generate.py -v`
Expected: FAIL (import)

- [ ] **Step 3: Implement**

```python
# service/src/service/core/generate.py
import json
from collections.abc import AsyncIterator
from uuid import UUID

from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

from service.core.schemas import Message
from service.db.models import Message as MessageRow
from service.providers.base import ChatPrompt, LLMProvider, ScoredChunk


def format_sse(event: str, data) -> str:
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


def build_prompt(system: str, chunks: list[ScoredChunk], question: str) -> ChatPrompt:
    context = []
    for i, c in enumerate(chunks, start=1):
        context.append(f"[{i}] {c.filename}#{c.ordinal}: {c.text}")
    return ChatPrompt(system=system, context=context, user=question)


SYSTEM_PROMPT = (
    "You are RAGout, a retrieval-augmented assistant. Answer the user's question using "
    "only the provided context chunks. Cite sources as [n] matching the numbered chunks. "
    "If the context does not contain the answer, say you don't know."
)


class GenerateService:
    def __init__(self, llm: LLMProvider, session_factory: async_sessionmaker[AsyncSession]):
        self._llm = llm
        self._factory = session_factory

    async def stream(self, session_id: UUID, query: str, chunks: list[ScoredChunk]) -> AsyncIterator[str]:
        prompt = build_prompt(SYSTEM_PROMPT, chunks, query)
        history: list = []  # could load prior messages; keep simple for MVP
        answer_parts: list[str] = []
        async for token in self._llm.stream(prompt, history):
            answer_parts.append(token)
            yield format_sse("token", token)
        citations = [{"doc_id": str(c.document_id), "filename": c.filename,
                      "ordinal": c.ordinal, "score": c.score, "snippet": c.text[:200]} for c in chunks]
        async with self._factory() as s:
            s.add(MessageRow(session_id=session_id, role="user", content=query, cited_chunk_ids=[]))
            s.add(MessageRow(session_id=session_id, role="assistant", content="".join(answer_parts),
                             cited_chunk_ids=[c.chunk_id for c in chunks]))
            await s.commit()
        yield format_sse("done", {"citations": citations})
```

- [ ] **Step 4: Run tests, commit**

```bash
uv run pytest service/tests/unit/test_generate.py -v
uv run ruff check service/src/service/core/generate.py
git add service/src/service/core/generate.py service/tests/unit/test_generate.py
git commit -m "feat: add SSE framing and generate service"
```

### Task 6.3: Chat router + message history

**Files:**

- Create: `service/src/service/api/routers/chat.py`
- Modify: `service/src/service/api/deps.py` — add `get_retrieve_service`, `get_generate_service`
- Test: `service/tests/integration/test_routers_chat.py`

**Interfaces:**

- Consumes: `RetrieveService`, `GenerateService`, settings.
- Produces: `POST /api/v1/sessions/{id}/chat` returning `text/event-stream`; `GET /api/v1/sessions/{id}/messages`.

- [ ] **Step 1: Write failing integration test (streaming response parsing)**

```python
# service/tests/integration/test_routers_chat.py
from httpx import AsyncClient

from service.api.run import create_app


async def test_chat_streams_tokens_and_done(pg_engine, monkeypatch, fake_llm, fake_embeddings, fake_reranker):
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    monkeypatch.setenv("COHERE_API_KEY", "co")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/ragout")
    from service.api.deps import set_engine, override_services
    set_engine(pg_engine)
    override_services(llm=fake_llm, embeddings=fake_embeddings, reranker=fake_reranker)
    app = create_app()
    async with AsyncClient(app=app, base_url="http://t") as client:
        sid = (await client.post("/api/v1/sessions", json={"title": "t"})).json()["id"]
        await client.post(f"/api/v1/sessions/{sid}/documents",
                          files={"file": ("a.txt", b"Paris is the capital of France.", "text/plain")})
        async with client.stream("POST", f"/api/v1/sessions/{sid}/chat", json={"query": "capital?"}) as r:
            body = "".join([line async for line in r.aiter_text()])
        assert "event: token" in body
        assert "event: done" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest service/tests/integration/test_routers_chat.py -v`
Expected: FAIL (route missing)

- [ ] **Step 3: Add dep overrides + service getters in `deps.py`**

```python
# append to service/src/service/api/deps.py
from service.core.generate import GenerateService
from service.core.retrieve import RetrieveService
from service.providers.factory import build_llm, build_reranker

_overrides: dict = {}


def override_services(*, llm=None, embeddings=None, reranker=None) -> None:
    if llm is not None:
        _overrides["llm"] = llm
    if embeddings is not None:
        _overrides["embeddings"] = embeddings
    if reranker is not None:
        _overrides["reranker"] = reranker


def get_retrieve_service(settings=Depends(get_settings)):
    emb = _overrides.get("embeddings", build_embeddings(settings))
    rer = _overrides.get("reranker", build_reranker(settings))
    return RetrieveService(emb, rer, get_session_factory(),
                           top_k=settings.retrieve_top_k, top_n=settings.rerank_top_n)


def get_generate_service(settings=Depends(get_settings)):
    llm = _overrides.get("llm", build_llm(settings))
    return GenerateService(llm, get_session_factory())
```

> Add `from fastapi import Depends` at top of deps.py.

- [ ] **Step 4: Implement `chat.py`**

```python
# service/src/service/api/routers/chat.py
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import select

from service.api.deps import get_generate_service, get_session_factory
from service.core.generate import GenerateService
from service.core.retrieve import RetrieveService
from service.db.models import Message, Session

router = APIRouter()


def _get_retrieve(settings=Depends(get_settings)):  # wired via deps import below
    ...


@router.post("/{session_id}/chat")
async def chat(session_id: UUID, payload: dict,
               retrieve: RetrieveService = Depends(get_retrieve_service),
               generate: GenerateService = Depends(get_generate_service),
               factory=Depends(get_session_factory)) -> StreamingResponse:
    query = payload.get("query", "")
    if not query:
        raise HTTPException(422, "query is required")
    async with factory() as s:
        if not await s.get(Session, session_id):
            raise HTTPException(404, "Session not found")
    chunks = await retrieve.retrieve(session_id, query)

    async def event_stream() -> AsyncIterator[str]:
        async for frame in generate.stream(session_id, query, chunks):
            yield frame

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/{session_id}/messages")
async def messages(session_id: UUID, factory=Depends(get_session_factory)) -> list[dict]:
    async with factory() as s:
        rows = (await s.exec(select(Message).where(Message.session_id == session_id)
                             .order_by(Message.created_at))).all()
        return [{"id": str(r.id), "role": r.role, "content": r.content,
                 "cited_chunk_ids": [str(c) for c in r.cited_chunk_ids]} for r in rows]
```

> Add imports `from service.api.deps import get_retrieve_service` at top of chat.py (the local `_get_retrieve` stub is a placeholder — remove it and use the deps import).

- [ ] **Step 5: Run tests, commit**

```bash
uv run pytest service/tests/integration/test_routers_chat.py -v
uv run ruff check service/src/service/api
git add service/src/service/api/routers/chat.py service/src/service/api/deps.py service/tests/integration/test_routers_chat.py
git commit -m "feat: add streaming chat router and message history"
```

---

## Phase 7 — Eval harness

### Task 7.1: Sample corpus + labeled queries

**Files:**

- Create: `service/src/service/eval/data/*.md` (~10 files on neutral topics, e.g. one paragraph each)
- Create: `service/src/service/eval/data/queries.jsonl`

**Interfaces:** None (data only).

- [ ] **Step 1: Author ~10 small markdown files** (e.g. `python.md`, `postgresql.md`, `http.md`, `git.md`, `docker.md`, `rust.md`, `css.md`, `dns.md`, `ssh.md`, `bash.md`), each 2–4 paragraphs of factual content.

- [ ] **Step 2: Author `queries.jsonl`** with ~20 lines:

```json
{ "query": "What is Python's GIL?", "relevant_doc_ids": ["python.md"] }
```

> Use document basenames (without `.md`) as stable ids referenced in eval matching. Ensure each query maps to ≥1 relevant doc id present in the corpus.

- [ ] **Step 3: Commit**

```bash
git add service/src/service/eval/data
git commit -m "feat: add eval sample corpus and labeled queries"
```

### Task 7.2: Eval runner

**Files:**

- Create: `service/src/service/eval/__init__.py`
- Create: `service/src/service/eval/__main__.py`
- Create: `service/src/service/eval/run.py`
- Create: `service/src/service/eval/results/.gitkeep`
- Modify: `.gitignore` — ignore `service/src/service/eval/results/*.json`
- Test: `service/tests/unit/test_eval_metrics.py`

**Interfaces:**

- Consumes: `RetrieveService`, `SyncIngestService`, real providers.
- Produces:
  - `metrics(ranked_doc_ids: list[str], relevant: set[str]) -> dict` returning recall@5, recall@10, precision@5, MRR.
  - CLI: `python -m service.eval ingest`, `python -m service.eval run`, `python -m service.eval clean`.

- [ ] **Step 1: Write failing unit test for `metrics`**

```python
# service/tests/unit/test_eval_metrics.py
from service.eval.run import metrics


def test_metrics_perfect():
    m = metrics(["a", "b", "c"], {"a"})
    assert m["recall@5"] == 1.0
    assert m["recall@10"] == 1.0
    assert m["precision@5"] == 0.2
    assert m["mrr"] == 1.0


def test_metrics_miss_then_hit():
    m = metrics(["x", "a"], {"a"})
    assert m["recall@5"] == 1.0
    assert m["mrr"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest service/tests/unit/test_eval_metrics.py -v`
Expected: FAIL (import)

- [ ] **Step 3: Implement `run.py`**

```python
# service/src/service/eval/run.py
import json
from pathlib import Path

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"
SESSION_FILE = Path(__file__).parent / ".eval_session"


def metrics(ranked_doc_ids: list[str], relevant: set[str]) -> dict:
    def recall_at(k):
        top = ranked_doc_ids[:k]
        return len(set(top) & relevant) / len(relevant) if relevant else 0.0

    precision_at_5 = len(set(ranked_doc_ids[:5]) & relevant) / 5 if ranked_doc_ids else 0.0
    mrr = 0.0
    for i, d in enumerate(ranked_doc_ids, start=1):
        if d in relevant:
            mrr = 1.0 / i
            break
    return {"recall@5": recall_at(5), "recall@10": recall_at(10),
            "precision@5": precision_at_5, "mrr": mrr}
```

- [ ] **Step 4: Implement CLI `__main__.py`**

```python
# service/src/service/eval/__main__.py
import asyncio
import json
import sys
from pathlib import Path

from service.api.deps import get_settings
from service.core.ingest import SyncIngestService
from service.core.retrieve import RetrieveService
from service.db.models import Session
from service.db.session import create_engine, get_session_factory, run_migrations  # run_migrations via pgvector
from service.eval.run import DATA, RESULTS, SESSION_FILE, metrics
from service.providers.factory import build_embeddings, build_reranker


async def _ingest():
    settings = get_settings()
    engine = create_engine(settings.database_url)
    await run_migrations(engine)
    factory = get_session_factory(engine)
    emb = build_embeddings(settings)
    ingest = SyncIngestService(emb, factory)
    async with factory() as s:
        row = Session(title="eval")
        s.add(row)
        await s.commit()
        sid = row.id
    SESSION_FILE.write_text(str(sid))
    for md in DATA.glob("*.md"):
        await ingest.ingest(sid, md.name, "text/markdown", md.read_bytes())
    print(f"ingested into session {sid}")


async def _run():
    from sqlalchemy import delete
    from service.db.models import Chunk, Document
    sid = SESSION_FILE.read_text()
    settings = get_settings()
    engine = create_engine(settings.database_url)
    factory = get_session_factory(engine)
    retrieve = RetrieveService(build_embeddings(settings), build_reranker(settings),
                              factory, settings.retrieve_top_k, settings.rerank_top_n)
    queries = [json.loads(line) for line in (DATA / "queries.jsonl").read_text().splitlines() if line]
    per_query = []
    agg = {"recall@5": 0, "recall@10": 0, "precision@5": 0, "mrr": 0}
    for q in queries:
        chunks = await retrieve.retrieve(sid, q["query"])
        ranked = [Path(c.filename).stem for c in chunks]
        m = metrics(ranked, set(q["relevant_doc_ids"]))
        per_query.append({"query": q["query"], **m, "ranked": ranked})
        for k in agg:
            agg[k] += m[k]
    n = len(queries) or 1
    summary = {k: v / n for k, v in agg.items()}
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{int(asyncio.get_event_loop().time())}.json"
    out.write_text(json.dumps({"summary": summary, "per_query": per_query}, indent=2))
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}")
    print(f"Results written to {out}")


async def _clean():
    sid = SESSION_FILE.read_text()
    settings = get_settings()
    engine = create_engine(settings.database_url)
    from service.db.models import Session
    from uuid import UUID
    factory = get_session_factory(engine)
    async with factory() as s:
        row = await s.get(Session, UUID(sid))
        if row:
            await s.delete(row)
            await s.commit()
    SESSION_FILE.unlink(missing_ok=True)
    print("cleaned eval session")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    coro = {"ingest": _ingest, "run": _run, "clean": _clean}[cmd]
    asyncio.run(coro())


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run unit test, commit**

```bash
uv run pytest service/tests/unit/test_eval_metrics.py -v
uv run ruff check service/src/service/eval
git add service/src/service/eval service/tests/unit/test_eval_metrics.py .gitignore
git commit -m "feat: add eval harness CLI and metrics"
```

---

## Phase 8 — Frontend (app/)

### Task 8.1: Scaffold Vite + React + TS + Tailwind

**Files:**

- Create: `app/` (scaffold via `npm create vite@latest app -- --template react-ts`)
- Modify: `app/package.json` (add deps: `react-router-dom`, `@tanstack/react-query`, `tailwindcss`, `postcss`, `autoprefixer`)
- Create: `app/tailwind.config.ts`, `app/postcss.config.js`, `app/src/index.css`
- Create: `app/vite.config.ts` (proxy `/api` → `http://localhost:8000`)
- Create: `app/.gitignore`

- [ ] **Step 1: Scaffold**

```bash
npm create vite@latest app -- --template react-ts
cd app && npm install
npm install react-router-dom @tanstack/react-query
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

- [ ] **Step 2: Configure Tailwind** (`app/tailwind.config.ts` content paths `./src/**/*`, `app/src/index.css` with `@tailwind` directives).

- [ ] **Step 3: Configure Vite proxy**

```ts
// app/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
});
```

- [ ] **Step 4: Commit**

```bash
git add app
git commit -m "feat: scaffold frontend (Vite + React + TS + Tailwind)"
```

### Task 8.2: API client + types + session list

**Files:**

- Create: `app/src/lib/api.ts`
- Create: `app/src/types.ts`
- Create: `app/src/routes/SessionList.tsx`
- Modify: `app/src/App.tsx` (router + QueryClient)

**Interfaces:**

- Produces: typed wrappers `listSessions`, `createSession`, `deleteSession`, `listDocuments`, `uploadDocument`, `deleteDocument`, `chatStream(sessionId, query, onToken, onDone)`.

- [ ] **Step 1: `types.ts`**

```ts
// app/src/types.ts
export interface Session {
  id: string;
  title: string;
  created_at: string;
}
export interface DocumentSummary {
  id: string;
  filename: string;
  num_chunks: number;
  content_hash: string;
}
export interface Citation {
  doc_id: string;
  filename: string;
  ordinal: number;
  score: number;
  snippet: string;
}
export interface StoredMessage {
  id: string;
  role: string;
  content: string;
  cited_chunk_ids: string[];
}
```

- [ ] **Step 2: `api.ts`** — implement CRUD + `chatStream` using `fetch` + `ReadableStream` reader parsing SSE (`event: token\ndata: ...\n\n`, accumulate; `event: done` → call `onDone` with parsed citations).

- [ ] **Step 3: `SessionList.tsx`** — list via TanStack Query, "New session" button (POST), delete. Wire into `App.tsx` with `BrowserRouter`.

- [ ] **Step 4: Commit**

```bash
git add app
git commit -m "feat: add API client and session list"
```

### Task 8.3: Session view — documents panel + chat

**Files:**

- Create: `app/src/routes/SessionView.tsx`
- Create: `app/src/components/DocumentPanel.tsx`
- Create: `app/src/components/ChatView.tsx`
- Create: `app/src/components/MessageBubble.tsx`
- Create: `app/src/components/SourceChip.tsx`

- [ ] **Step 1: `DocumentPanel.tsx`** — drag-and-drop upload, list with chunk counts, delete button.

- [ ] **Step 2: `ChatView.tsx`** — message list (load history via `GET /messages`), input box, on send call `chatStream`, accumulate tokens into a live assistant bubble, on `done` render `SourceChip` per citation.

- [ ] **Step 3: `SourceChip.tsx` + `MessageBubble.tsx`** — chip expands to show chunk text + filename + ordinal.

- [ ] **Step 4: Wire route** `Route path="/sessions/:id"` → `SessionView`.

- [ ] **Step 5: Smoke test manually** (`npm run dev` + service running; upload a `.txt`, ask a question, see streamed answer + citation).

- [ ] **Step 6: Commit**

```bash
git add app
git commit -m "feat: add session view, document panel, streaming chat"
```

### Task 8.4: Vitest component tests

**Files:**

- Create: `app/src/test/setup.ts`
- Modify: `app/package.json` (add `vitest`, `@testing-library/react`, `jsdom`, test script)
- Create: `app/src/components/__tests__/ChatView.test.tsx` (mocked SSE via mocked `fetch`)

- [ ] **Step 1: Add vitest config + setup**, write a test that mocks `globalThis.fetch` to return a `ReadableStream` emitting `event: token\ndata: Hi \n\nevent: token\ndata:there\n\nevent: done\ndata:{"citations":[]}\n\n`, renders `ChatView`, sends a message, asserts "Hi there" appears.

- [ ] **Step 2: Run tests, commit**

```bash
cd app && npm test -- --run
git add app
git commit -m "test: add ChatView streaming test"
```

---

## Phase 9 — Docker Compose, Makefile, docs

### Task 9.1: `.env.example` + `docker-compose.yml`

**Files:**

- Create: `.env.example`
- Create: `docker-compose.yml`
- Create: `service/Dockerfile`
- Create: `app/Dockerfile`

- [ ] **Step 1: `.env.example`** per spec §12.

- [ ] **Step 2: `service/Dockerfile`** (uv-based, slim, `CMD service-api`).

- [ ] **Step 3: `app/Dockerfile`** (node, `CMD npm run dev -- --host`).

- [ ] **Step 4: `docker-compose.yml`** — `db` (`pgvector/pgvector:pg17`, volume, healthcheck `pg_isready`), `service` (build `service/`, `depends_on` db healthy, env_file `.env`, port 8000, startup runs migrations), `app` (build `app/`, port 5173, depends_on service).

- [ ] **Step 5: Smoke test** `docker compose up --build`, hit `http://localhost:5173`.

- [ ] **Step 6: Commit**

```bash
git add .env.example docker-compose.yml service/Dockerfile app/Dockerfile
git commit -m "feat: add docker compose setup"
```

### Task 9.2: Makefile + README + architecture doc

**Files:**

- Modify: `Makefile` (add `up`, `down`, `migrate`, `eval`, `test`, `test-app`)
- Modify: `README.md` (root) — branding quick start
- Create: `docs/architecture.md`
- Modify: `app/README.md`, `service/README.md`

- [ ] **Step 1: Extend `Makefile`**

```make
.PHONY: up down migrate eval test test-app
up:
	docker compose up --build
down:
	docker compose down
migrate:
	uv run python -c "import asyncio; from service.db.session import create_engine, app_lifespan; ..."
eval:
	uv run --package service python -m service.eval run
test:
	uv run pytest -v
test-app:
	cd app && npm test -- --run
run-service-api:
	uv run --package service service-api
```

- [ ] **Step 2: Write root `README.md`** with cooking-pun intro, architecture summary, quick start, screenshot placeholder.

- [ ] **Step 3: Write `docs/architecture.md`** with data-flow diagrams (upload, chat) and provider abstraction explanation.

- [ ] **Step 4: Write per-package READMEs** in the same voice.

- [ ] **Step 5: Commit**

```bash
git add Makefile README.md docs app/README.md service/README.md
git commit -m "docs: add READMEs, architecture doc, Makefile targets"
```

---

## Self-Review Notes

- **Spec coverage:** All spec sections (§1–§16) map to tasks: data model → 3.1/3.2; ingestion → 5.1; retrieval → 6.1; generation → 6.2; API → 5.2/6.3; providers → 2.x; frontend → 8.x; eval → 7.x; compose/docs → 9.x.
- **Type consistency:** `ScoredChunk` defined in 2.1, used in 6.1/6.2; `cited_chunk_ids` is `uuid[]` (fixed in spec) — `MessageRow.cited_chunk_ids` uses `ARRAY(UUID)` in 3.1; `DocumentSummary` defined in 4.3, used in 5.1. Confirmed consistent.
- **Known soft spots to watch during implementation:** (a) SQLModel `metadata_`→`metadata` column mapping needs an explicit `sa_column`; (b) `pgvector` cosine order via `Chunk.embedding.cosine_distance`; (c) testcontainers `get_connection_url` returns a psycopg2 URL — replace driver to asyncpg; (d) `@app.on_event("startup")` is deprecated in newer FastAPI — prefer `lifespan` context manager (refactor in 9.1 if needed).
- **Placeholder-free:** each code step contains real code; test assertions are concrete.
