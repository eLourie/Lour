"""
app/core/config.py

Typed twelve-factor configuration via Pydantic Settings.
All backing-service addresses come from environment / .env — never hardcoded.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Enums (keep in sync with .env.example comments)


class AppEnv(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class AuthMode(StrEnum):
    APIKEY = "apikey"  # core
    JWT = "jwt"  # showcase


class LLMProvider(StrEnum):
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    VLLM = "vllm"


class RerankerMode(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"
    NONE = "none"


class WebSearchProvider(StrEnum):
    TAVILY = "tavily"
    SEARXNG = "searxng"


class TelemetryBackend(StrEnum):
    LANGFUSE_CLOUD = "langfuse_cloud"
    LANGFUSE_SELFHOST = "langfuse_selfhost"
    NONE = "none"


class DeployProfile(StrEnum):
    SOLO = "solo"
    SPLIT = "split"
    OFFLOADED = "offloaded"


# Settings groups (nested for clarity)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    env: AppEnv = AppEnv.DEVELOPMENT
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # JSON is the always-on observability baseline (works offline, machine-readable).
    # Set APP_LOG_FORMAT=console for pretty local-dev output.
    log_format: Literal["json", "console"] = "json"
    api_key: str = Field(default="changeme-user", description="User API key")
    admin_api_key: str = Field(default="changeme-admin", description="Admin API key")
    auth_mode: AuthMode = AuthMode.APIKEY


class LLMSettings(BaseSettings):
    provider: LLMProvider = LLMProvider.OLLAMA
    main_model: str = "qwen3:14b"
    fast_model: str = ""  # empty = main model routes itself
    embed_model: str = "bge-m3"

    # Filled from separate ANTHROPIC_* / OPENAI_* env vars
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_", env_file=".env", extra="ignore")

    base_url: str = "http://host.docker.internal:11434"
    timeout_s: int = 120


class RerankerSettings(BaseSettings):
    mode: RerankerMode = RerankerMode.LOCAL
    model: str = "bge-reranker-v2-m3"
    base_url: str = "http://host.docker.internal:8081"
    lazy_load: bool = True
    cohere_api_key: str = Field(default="", alias="COHERE_API_KEY")

    model_config = SettingsConfigDict(
        env_prefix="RERANKER_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )


class PostgresSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env", extra="ignore")

    host: str = "localhost"
    port: int = 5432
    db: str = "agent"
    user: str = "agent"
    password: str = "changeme"
    checkpoint_db: str = "agent_checkpoint"

    @property
    def dsn(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @property
    def checkpoint_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.checkpoint_db}"
        )

    @property
    def dsn_sync(self) -> str:
        """Used by Alembic (sync driver)."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")

    host: str = "localhost"
    port: int = 6379
    db_cache: int = 0
    db_memory: int = 1
    db_ratelimit: int = 2

    def url(self, db: int) -> str:
        return f"redis://{self.host}:{self.port}/{db}"


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QDRANT_", env_file=".env", extra="ignore")

    host: str = "localhost"
    port: int = 6333
    collection_docs: str = "documents"
    collection_memory: str = "memories"


class TelemetrySettings(BaseSettings):
    backend: TelemetryBackend = Field(
        default=TelemetryBackend.LANGFUSE_CLOUD,
        alias="TELEMETRY_BACKEND",
    )
    host: str = "https://cloud.langfuse.com"
    public_key: str = ""
    secret_key: str = ""

    model_config = SettingsConfigDict(
        env_prefix="LANGFUSE_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    max_iterations: int = 10
    max_tool_calls: int = 20
    budget_tokens: int = 100_000


class MemorySettings(BaseSettings):
    """Three-layer memory: short-term (Redis), long-term (Qdrant), episodic (PG)."""

    model_config = SettingsConfigDict(env_prefix="MEMORY_", env_file=".env", extra="ignore")

    # Short-term: messages kept verbatim in the sliding window before the tail
    # is summarised. TTL bounds how long an idle session's window survives.
    short_term_window: int = 20
    short_term_ttl_s: int = 60 * 60 * 24  # 1 day

    # Long-term recall
    long_term_top_k: int = 5
    # Over-fetch factor: pull this * top_k dense candidates, then re-score.
    candidate_multiplier: int = 6

    # Combined score = alpha*cosine + beta*recency + gamma*importance.
    # Weights need not sum to 1 — ordering is what matters.
    score_alpha: float = 0.6
    score_beta: float = 0.2
    score_gamma: float = 0.2
    # Recency decays by half every this many hours (exponential).
    recency_half_life_h: float = 72.0

    # Consolidation (APScheduler background job)
    consolidation_enabled: bool = True
    consolidation_interval_s: int = 3600
    # A candidate fact whose nearest existing memory scores >= this cosine is
    # treated as a duplicate and skipped.
    dedup_threshold: float = 0.9
    # Minimum importance for an extracted fact to be persisted to long-term.
    min_importance: float = 0.3


class SandboxSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SANDBOX_", env_file=".env", extra="ignore")

    memory_mb: int = 512
    cpu_quota: float = 1.0
    timeout_s: int = 30
    pids_limit: int = 128
    python_image: str = "python:3.12-slim"
    node_image: str = "node:20-slim"


class ToolsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TOOLS_", env_file=".env", extra="ignore")

    # filesystem tool is confined to this directory (path-traversal guarded)
    workspace_dir: str = "./workspace"
    # http_request tool only reaches these hosts (comma-separated in env)
    http_allowlist: list[str] = Field(default_factory=list)
    # result cache TTL for idempotent (non-side-effecting) tools
    cache_ttl_s: int = 300

    @field_validator("http_allowlist", mode="before")
    @classmethod
    def _split_allowlist(cls, v: object) -> object:
        # Accept a comma-separated string from env; pass lists through untouched.
        if isinstance(v, str):
            return [h.strip() for h in v.split(",") if h.strip()]
        return v


class McpSettings(BaseSettings):
    """External MCP servers this instance connects to as a client (ADR-006)."""

    model_config = SettingsConfigDict(env_prefix="MCP_", env_file=".env", extra="ignore")

    # JSON mapping of {name: {command, args, env}} for stdio MCP servers.
    # Empty by default — MCP client stays dormant until configured.
    servers_json: str = ""

    def servers(self) -> dict[str, dict[str, Any]]:
        """Parse the configured stdio servers; return {} when unset or invalid."""
        if not self.servers_json.strip():
            return {}
        import json

        parsed: dict[str, dict[str, Any]] = json.loads(self.servers_json)
        return parsed


# Root settings (composes all groups)


class Settings(BaseSettings):
    """
    Single entry-point for all configuration.

    Usage:
        from app.core.config import get_settings
        settings = get_settings()
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deploy_profile: DeployProfile = DeployProfile.SPLIT
    sparse_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    web_search_provider: WebSearchProvider = WebSearchProvider.TAVILY
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    searxng_base_url: str = Field(default="http://localhost:8080", alias="SEARXNG_BASE_URL")

    # Nested groups — each reads its own prefix from the same .env
    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    tools: ToolsSettings = Field(default_factory=ToolsSettings)
    mcp: McpSettings = Field(default_factory=McpSettings)

    @field_validator("deploy_profile", mode="before")
    @classmethod
    def _coerce_deploy_profile(cls, v: object) -> object:
        if isinstance(v, str):
            return v.lower()
        return v

    @property
    def is_production(self) -> bool:
        return self.app.env == AppEnv.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.app.env == AppEnv.DEVELOPMENT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton. Call once per process."""
    return Settings()
