from __future__ import annotations

import pytest

from fakes import FakeEmbeddings, FakeLLM, FakeReranker


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def fake_reranker():
    return FakeReranker()
