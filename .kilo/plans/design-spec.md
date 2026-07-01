# RAGout — Design Spec

> A tasty RAG stew. Toss in your ingredients (documents), let them simmer
> (chunk → embed → rerank), and serve (answers with citations).
>
> RAGout is a portfolio project: an upload-on-demand Retrieval-Augmented
> Generation service plus a chat frontend. The name plays on _ragoût_ (French
> for "stew"); the cooking metaphor lives in branding and docs, while the
> corpus content stays neutral and user-supplied.

## 1. Goals & non-goals

**Goals**

- Let a reviewer `docker compose up`, add API keys, upload a few documents,
  and chat with them — answers cite source chunks.
- Demonstrate a clean, layered, pluggable architecture suitable as a
  portfolio showcase (provider abstraction, reranking, an eval harness).
- Be easy to run locally end-to-end with one command.

**Non-goals (this iteration)**

- User accounts / multi-tenant auth.
- Live hosted deployment (local-run only for now).
- Background/async ingestion with a job queue (designed _for_ it, but the
  MVP uses a synchronous ingest implementation behind a swappable interface).
- GPU / local models (all hosted APIs by default).

## 2. Stack decisions (locked)

| Concern          | Choice                                                                       |
| ---------------- | ---------------------------------------------------------------------------- |
| Backend          | FastAPI (existing `service/` uv workspace member)                            |
| Language         | Python 3.14, uv workspace, ruff                                              |
| DB + vectors     | Postgres with pgvector (HNSW, cosine)                                        |
| ORM              | SQLModel                                                                     |
| LLM + embeddings | Pluggable provider; default **OpenAI** (gpt-4o-mini, text-embedding-3-small) |
| Reranker         | Pluggable provider; default **Cohere Rerank** (rerank-english-v3.0)          |
| Frontend         | Vite + React + TypeScript, separate `app/` dir                               |
| Frontend styling | Tailwind CSS                                                                 |
| Frontend data    | TanStack Query (sessions/docs) + local state (chat)                          |
| Streaming        | Server-Sent Events (SSE) over HTTP POST                                      |
| Delivery         | Docker Compose (db + service + app)                                          |
| Tests            | pytest + testcontainers (service); Vitest (app)                              |

## 3. Repository layout

```
ragout/
  app/                          # Vite + React + TS frontend (new)
    src/
      routes/
      components/
      lib/
      App.tsx
      main.tsx
    index.html
    package.json
    vite.config.ts
    tsconfig.json
    tailwind.config.ts
  service/                      # FastAPI backend (existing uv member)
    src/service/
      api/
        routers/
          health.py             # existing on feat/api-init
          sessions.py
          documents.py
          chat.py
        deps.py                 # FastAPI dependency wiring (DI)
        run.py                  # existing app factory + entrypoint
      core/
        parsing.py              # PDF / MD / TXT → text
        chunking.py             # markdown-aware + recursive char split
        ingest.py               # IngestService interface + SyncIngestService
        retrieve.py             # vector ANN → Cohere rerank
        generate.py             # prompt assembly + streaming LLM call
      providers/
        base.py                 # Protocol definitions
        llm.py                  # LLMProvider + OpenAILLM
        embeddings.py           # EmbeddingsProvider + OpenAIEmbeddings
        reranker.py            # Reranker + CohereReranker
        factory.py             # env-driven provider selection
      db/
        models.py               # SQLModel: Session, Document, Chunk, Message
        session.py              # async engine + session factory
        pgvector.py             # vector column / index helpers
        migrations/             # SQL init scripts (run on startup or via make migrate)
      config.py                 # pydantic-settings Settings
      eval/
        __main__.py             # python -m service.eval
        run.py
        data/                   # bundled sample corpus + labeled query set
        results/                # JSON outputs (gitignored)
    tests/
      unit/
      integration/
  docker-compose.yml            # db + service + app
  Makefile                      # up, down, run-service-api (existing), eval, test, migrate
  docs/
    architecture.md
    assets/                      # diagrams
  .env.example
  pyproject.toml                # workspace root (existing)
```

`app/` is a JS workspace at repo root (not a uv member). The existing
`feat/api-init` branch (health router, app factory, `service-api` script,
Makefile target) is merged and extended — not replaced.

## 4. Data model (Postgres + pgvector)

All tables live in a single Postgres database; `session_id` scopes every
query so each chat session sees only its own uploaded documents.

- `sessions`
  - `id uuid pk`, `title text`, `created_at timestamptz`
- `documents`
  - `id uuid pk`, `session_id uuid fk → sessions(id) on delete cascade`
  - `filename text`, `mime text`, `content_hash text` (sha256)
  - `num_chunks int`, `created_at timestamptz`
- `chunks`
  - `id uuid pk`, `document_id uuid fk → documents(id) on delete cascade`
  - `ordinal int`, `text text`, `metadata jsonb`
  - `embedding vector(N)` (N = embedding dim, e.g. 1536 for text-embedding-3-small)
  - HNSW index on `embedding` (cosine), plus a btree on `document_id`
- `messages`
  - `id uuid pk`, `session_id uuid fk → sessions(id) on delete cascade`
  - `role text` (`user` | `assistant`), `content text`
  - `cited_chunk_ids uuid[]`, `created_at timestamptz`

Cascades ensure deleting a session or document removes its chunks/messages.

## 5. Ingestion pipeline (synchronous now, swappable later)

Interface:

```python
class IngestService(Protocol):
    async def ingest(self, session_id: UUID, filename: str, raw: bytes) -> DocumentSummary: ...
```

Steps (the default `SyncIngestService`):

1. Enforce `MAX_UPLOAD_BYTES` (default 10 MB) before reading fully.
2. Detect MIME; parse:
   - PDF → `pypdf` (extract text per page, join with form feeds as boundaries)
   - Markdown → `markdown-it-py` to plain text (strip markup, keep headings as delimiters)
   - `text/plain` → passthrough
3. Chunk via `core/chunking.py`:
   - Markdown-aware splitter: split on `#`/`##`/`###` headings first; within a section, recursive char split (~800 chars, ~100 overlap).
   - PDF/TXT: recursive char split directly.
   - Each chunk carries `metadata`: `{source: filename, ordinal, heading, page?}`.
4. Batch embed via `EmbeddingsProvider.embed(chunk_texts)`.
5. Persist `Document` row + `Chunk` rows with embeddings in one DB transaction.
6. Return `DocumentSummary {id, filename, num_chunks, content_hash}`.

The interface lets a future `BackgroundTaskIngestService` (Approach B) replace
this with job polling without touching routers.

## 6. Retrieval

```python
class RetrieveService:
    async def retrieve(self, session_id: UUID, query: str, top_k=20, top_n=5) -> list[ScoredChunk]: ...
```

1. Embed query via `EmbeddingsProvider`.
2. ANN search in pgvector (cosine) over `chunks` scoped to `session_id`, return `top_k` (default 20).
3. `Reranker.rerank(query, candidate_texts, top_n)` (Cohere) → top 5.
4. Return `ScoredChunk {chunk_id, document_id, filename, ordinal, text, score, metadata}`.

`top_k` / `top_n` are configurable via `Settings`.

## 7. Generation + streaming

`core/generate.py`:

1. Build prompt:
   - System: functional assistant with RAGout's cooking-pun voice; instruct to answer only from provided chunks and to cite by `[n]`.
   - Context: retrieved chunks rendered as numbered `[n] {filename}#{ordinal}: {text}`.
   - User question.
2. Call `LLMProvider.stream(prompt, history)` and forward each token chunk as an SSE `event: token` / `data: "..."` frame.
3. After the stream completes, emit `event: done` / `data: {citations: [...]}` where each citation is `{doc_id, filename, ordinal, score, snippet}`.
4. Persist `messages` row pair (user question + assistant answer + `cited_chunk_ids`).

## 8. API surface (`/api/v1`)

| Method | Path                                | Purpose                         |
| ------ | ----------------------------------- | ------------------------------- |
| GET    | `/health`                           | Health (existing)               |
| POST   | `/sessions`                         | Create session `{title?}`       |
| GET    | `/sessions`                         | List sessions                   |
| DELETE | `/sessions/{id}`                    | Delete session + cascade        |
| POST   | `/sessions/{id}/documents`          | Multipart upload → doc summary  |
| GET    | `/sessions/{id}/documents`          | List session's documents        |
| DELETE | `/sessions/{id}/documents/{doc_id}` | Delete a document + its chunks  |
| POST   | `/sessions/{id}/chat`               | SSE stream: tokens + done event |
| GET    | `/sessions/{id}/messages`           | Replay persisted chat history   |

All endpoints return JSON; errors use FastAPI's standard problem shape
(`{detail}`) with explicit status codes (400 upload too large, 404 unknown
session/doc, 422 parse failure, 502 provider error).

## 9. Provider abstraction

`providers/base.py` defines Protocols:

```python
class LLMProvider(Protocol):
    async def stream(self, prompt: ChatPrompt, history: list[Message]) -> AsyncIterator[str]: ...
    async def complete(self, prompt: ChatPrompt, history: list[Message]) -> str: ...

class EmbeddingsProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

class Reranker(Protocol):
    async def rerank(self, query: str, docs: list[str], top_n: int) -> list[ScoredText]: ...
```

`providers/factory.py` selects implementations from `Settings`:

- `LLM_PROVIDER` (default `openai`) → `OpenAILLM`
- `EMBEDDINGS_PROVIDER` (default `openai`) → `OpenAIEmbeddings`
- `RERANKER_PROVIDER` (default `cohere`) → `CohereReranker`

A local/stub implementation can be added later behind the same interfaces
(e.g. an `OllamaLLM`, a `sentence-transformers` reranker) with no caller
changes. Wiring is via FastAPI `Depends` in `api/deps.py`.

## 10. Frontend (`app/`)

- **Routes** (React Router):
  - `/` — session list + "New session" button.
  - `/sessions/:id` — chat view + document panel.
- **Components**:
  - `SessionSidebar` — list, create, switch, delete sessions.
  - `DocumentPanel` — drag-and-drop upload, list with chunk counts, delete.
  - `ChatView` — message list, input box, streaming render.
  - `MessageBubble` — user/assistant styling, assistant shows citation chips.
  - `SourceChip` — clickable → expands to show chunk text + filename + ordinal.
- **Streaming**: `fetch` POST to `/chat` with `Accept: text/event-stream`,
  read via `ReadableStream` reader, parse SSE frames manually (handles POST
  streaming, which `EventSource` cannot). Accumulate tokens into the live
  assistant bubble; on `done` event, render citation chips.
- **Data**: TanStack Query for `sessions` and `documents` CRUD; local React
  state for the active chat stream and persisted message history.
- **Styling**: Tailwind CSS, responsive, light/dark optional.
- **API client**: thin `lib/api.ts` with typed wrappers; base URL via Vite
  env (`VITE_API_BASE`), dev proxy to `http://localhost:8000`.

## 11. Eval harness (`python -m service.eval`)

- **Bundled data** in `service/eval/data/`:
  - A small sample corpus (~10–15 markdown files on neutral topics).
  - A labeled query set (`queries.jsonl`): ~20 entries of
    `{query, relevant_doc_ids: [...], relevant_chunk_ordinals: [...]}`.
- **Subcommands**:
  - `python -m service.eval ingest` — creates a fresh dedicated session,
    loads the sample corpus into it, and records the session id to a local
    file so `run` reuses it. Eval sessions are deleted on a `clean` subcommand
    so they never pollute user sessions.
  - `python -m service.eval run` — for each query, runs the real
    `RetrieveService` (ANN + rerank) and the `GenerateService`, then computes
    retrieval metrics and (optionally) answer faithfulness.
- **Metrics**: `recall@5`, `recall@10`, `precision@5`, `MRR`. Optional
  end-to-end answer faithfulness via a simple LLM-as-judge prompt (gated
  behind a flag, since it spends tokens).
- **Output**: a table to stdout + a JSON file in
  `service/eval/results/<timestamp>.json` (gitignored).
- **Reuse**: the harness imports the production `RetrieveService` /
  `GenerateService` so it exercises the real path, not a copy.

## 12. Config, secrets, compose

- `.env.example`:
  ```
  OPENAI_API_KEY=
  COHERE_API_KEY=
  DATABASE_URL=postgresql+asyncpg://ragout:ragout@db:5432/ragout
  LLM_PROVIDER=openai
  EMBEDDINGS_PROVIDER=openai
  RERANKER_PROVIDER=cohere
  MAX_UPLOAD_BYTES=10485760
  ```
- `config.py`: `pydantic-settings` `Settings` loaded from env, validated on
  startup; missing required keys fail fast with a clear message.
- `docker-compose.yml` services:
  - `db`: `pgvector/pgvector:pg17` image, named volume for data, healthcheck.
  - `service`: built from `service/`, `depends_on` db (healthy), port
    `8000:8000`, reads env from `.env`.
  - `app`: built from `app/`, runs Vite dev server, port `5173:5173`,
    proxies `/api` to `service:8000`.
- `Makefile` targets (extend existing):
  - `make up` / `make down` — compose lifecycle.
  - `make migrate` — apply DB init SQL (idempotent; CREATE EXTENSION IF NOT
    EXISTS vector, CREATE TABLE IF NOT EXISTS …).
  - `make run-service-api` — existing, kept.
  - `make eval` — run the eval harness against the running stack.
  - `make test` — pytest (service) + vitest (app).

## 13. Testing

- **Service** (`pytest`):
  - Unit: chunking (sizes/overlap/heading splits), parsing (PDF/MD/TXT
    fixtures), prompt assembly, SSE frame formatting, provider factory
    selection — no DB, no network.
  - Integration via `testcontainers` Postgres+pgvector: ingest a fixture
    doc → retrieve by a known query → assert relevant chunk is in top-N;
    cascade deletes; chat round-trip persistence.
  - Provider calls are mocked in tests (no real API spend); a
    `FakeLLM`/`FakeEmbeddings`/`FakeReranker` implement the Protocols for
    deterministic tests.
- **Frontend** (`vitest`):
  - Component tests for `DocumentPanel`, `ChatView` (mocked SSE),
    `SessionSidebar`.
  - Optional Playwright smoke: upload a doc, send a message, see a streamed
    answer with a citation chip.

## 14. Branding & docs

- **README.md** (root): cooking-pun voice intro, architecture summary, quick
  start (`cp .env.example .env`, add keys, `make up`), screenshot, link to
  `docs/architecture.md`. Example line: _"RAGout — toss in your ingredients
  (docs), let it simmer (chunk → embed → rerank), and serve (answers with
  citations)."_
- **docs/architecture.md**: request/data-flow diagrams (upload, chat,
  retrieve-rerank), provider abstraction explanation, how to add a new
  provider, how to extend to Approach B (background ingestion).
- Per-package READMEs (`app/README.md`, `service/README.md`) keep the same
  voice with package-specific run instructions.

## 15. Open / future work (explicitly out of MVP)

- Approach B: background ingestion + WebSocket streaming (interfaces already
  support the swap; see §5 and §7).
- Local providers via Ollama / sentence-transformers (interfaces support it;
  see §9).
- User accounts + multi-session ownership.
- Live hosted deployment.
- Cross-encoder reranker running locally (no Cohere dependency).

## 16. Decisions log

| #   | Decision                                      | Rationale                                          |
| --- | --------------------------------------------- | -------------------------------------------------- |
| 1   | Cooking puns in branding only, neutral corpus | User request; keeps the demo content flexible      |
| 2   | Upload-on-demand corpus                       | Self-contained demo, no fixed dataset to curate    |
| 3   | Pluggable providers, OpenAI/Cohere defaults   | Architecture flex for portfolio; cheap, well-known |
| 4   | pgvector via Postgres                         | Single DB, relational + vector, Docker-friendly    |
| 5   | Vite + React + TS in `app/`                   | Recognizable stack, lightweight, fast dev          |
| 6   | MVP = core + reranking + eval harness         | User choice; ambitious but bounded                 |
| 7   | Approach A now (sync ingest + SSE), B later   | Simplest reliable demo; interfaces enable the swap |
| 8   | Docker Compose local-run delivery             | No hosting cost; standard portfolio repo pattern   |

---
