"""
Application settings loaded from environment variables.
"""

from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["openai"]
EmbeddingsProviderName = Literal["openai"]
RerankerProviderName = Literal["cohere"]

DEFAULT_EMBEDDINGS_DIM = 1536


class Settings(BaseSettings):
    """Ragout configuration sourced from env vars and .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    cohere_api_key: str | None = None
    database_url: str

    llm_provider: LLMProviderName = "openai"
    embeddings_provider: EmbeddingsProviderName = "openai"
    reranker_provider: RerankerProviderName = "cohere"

    llm_model: str = "gpt-4o-mini"
    embeddings_model: str = "text-embedding-3-small"
    embeddings_dim: int = DEFAULT_EMBEDDINGS_DIM
    rerank_model: str = "rerank-english-v3.0"

    retrieve_top_k: int = 20
    rerank_top_n: int = 5

    # 10 MB limit
    max_upload_bytes: int = 10 * 1024 * 1024

    @model_validator(mode="after")
    def _check_provider_keys(self) -> "Settings":
        """Fail fast if a configured provider is missing its required API key."""
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.embeddings_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when EMBEDDINGS_PROVIDER=openai"
            )
        if self.reranker_provider == "cohere" and not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY is required when RERANKER_PROVIDER=cohere")
        return self
