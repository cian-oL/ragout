from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from fakes import FakeEmbeddings, FakeLLM, FakeReranker  # noqa: E402


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def fake_reranker():
    return FakeReranker()
