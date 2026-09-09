from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    app_env: str = Field(default="development")

    # LLM
    llm_provider: str = Field(default="ollama")
    llm_model: str = Field(default="qwen3:8b")
    openai_api_key: str = Field(default="")
    ollama_base_url: str = Field(default="http://localhost:11434")

    # Embeddings
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")

    # Reranker
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Qdrant
    qdrant_url: str | None = Field(default=None)
    qdrant_api_key: str | None = Field(default=None)
    qdrant_collection: str = Field(default="enterprise_rag")

    # Chunking (units: characters, using Python len())
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=80)

    # Retrieval
    dense_top_k: int = Field(default=10)
    bm25_top_k: int = Field(default=10)
    rrf_k: int = Field(default=60)
    rerank_top_n: int = Field(default=5)

    # Paths
    data_dir: str = Field(default="data")
    upload_dir: str = Field(default="data/uploads")
    index_dir: str = Field(default="data/index")
    qdrant_dir: str = Field(default="data/qdrant")
    chunk_registry_path: str = Field(default="data/index/chunks.jsonl")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
