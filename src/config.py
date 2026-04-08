"""Configuration management for CHEAP RAG system."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


# Load environment variables
load_dotenv()


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model_name: str
    device: str = "cuda"
    batch_size: int = 32
    cache_dir: str = "./models/embeddings"
    local_files_only: bool = False
    dimension: int = 768


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""

    type: str = "chroma"
    persist_directory: str = "./data/vector_db"
    collection_name: str = "cheap_metadata_v1"
    distance_metric: str = "cosine"
    metadata_fields: list[str] = Field(default_factory=list)
    index_config: dict[str, Any] = Field(default_factory=dict)


class OllamaConfig(BaseModel):
    """Ollama LLM configuration."""

    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5-coder:7b-instruct-q4_K_M"
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 1024
    context_length: int = 8192
    timeout_seconds: int = 60


class AnthropicConfig(BaseModel):
    """Anthropic Claude configuration."""

    api_key_env: str = "ANTHROPIC_API_KEY"
    model: str = "claude-sonnet-4-5-20250929"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout_seconds: int = 30
    track_costs: bool = True


class PromptConfig(BaseModel):
    """Prompt templates configuration."""

    system_message: str
    citation_format: str


class LLMConfig(BaseModel):
    """LLM configuration."""

    provider: str  # "ollama", "anthropic", or "hybrid"
    ollama: OllamaConfig | None = None
    anthropic: AnthropicConfig | None = None
    prompts: PromptConfig
    hybrid: dict[str, Any] | None = None


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""

    top_k: int = 5
    similarity_threshold: float = 0.3
    rerank: bool = False
    hybrid_search: dict[str, Any] = Field(default_factory=dict)


class ExtractorConfig(BaseModel):
    """Extractor configuration for a specific language."""

    enabled: bool = True
    file_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)


class DatabaseConnectionConfig(BaseModel):
    """Database connection parameters."""

    host: str = "localhost"
    port: int = 5432
    database: str = ""
    user: str = ""
    password: str = ""
    path: str | None = None  # SQLite only


class DatabaseConfig(BaseModel):
    """Configuration for a single database source."""

    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = True
    type: str  # "postgresql" or "sqlite"
    connection: DatabaseConnectionConfig
    schema_name: str = Field(default="public", alias="schema")
    include_tables: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class IndexingConfig(BaseModel):
    """Indexing pipeline configuration."""

    source_paths: list[str] = Field(default_factory=list)
    databases: dict[str, DatabaseConfig] = Field(default_factory=dict)
    extractors: dict[str, ExtractorConfig] = Field(default_factory=dict)
    batch_size: int = 100
    max_workers: int = 4


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "./logs/cheap-rag.log"
    log_api_calls: bool = False
    log_routing: bool = False


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=list)
    rate_limit: dict[str, Any] = Field(default_factory=dict)


class CostTrackingConfig(BaseModel):
    """Cost tracking configuration."""

    enabled: bool = False
    alert_threshold_usd: float = 10.0
    log_file: str = "./logs/costs.log"


class Config(BaseModel):
    """Complete system configuration."""

    embedding: EmbeddingConfig
    vectorstore: VectorStoreConfig
    llm: LLMConfig
    retrieval: RetrievalConfig
    indexing: IndexingConfig
    logging: LoggingConfig
    api: APIConfig
    cost_tracking: CostTrackingConfig | None = None
    routing_metrics: dict[str, Any] | None = None


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand ${VAR_NAME} references in string values.

    Args:
        obj: YAML-parsed object (dict, list, or scalar).

    Returns:
        Object with env var references replaced by their values.

    Raises:
        ValueError: If a referenced environment variable is not set.
    """
    if isinstance(obj, str):

        def _replace(m: re.Match[str]) -> str:
            var_name = m.group(1)
            value = os.getenv(var_name)
            if value is None:
                raise ValueError(
                    f"Environment variable '{var_name}' is not set. "
                    f"Add it to your .env file or shell environment."
                )
            return value

        return re.sub(r"\$\{([^}]+)\}", _replace, obj)
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in cast(dict[str, Any], obj).items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in cast(list[Any], obj)]
    return obj


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses CONFIG_PROFILE env var.

    Returns:
        Parsed configuration object.
    """
    if config_path is None:
        profile = os.getenv("CONFIG_PROFILE", "local")
        config_path = Path(__file__).parent.parent / "config" / f"{profile}.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config_dict = _expand_env_vars(config_dict)
    return Config(**config_dict)


def get_anthropic_api_key() -> str:
    """Get Anthropic API key from environment.

    Returns:
        API key string.

    Raises:
        ValueError: If API key is not set.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Required for Claude API mode. "
            "Copy .env.example to .env and add your API key."
        )
    return api_key
