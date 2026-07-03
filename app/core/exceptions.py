"""
app/core/exceptions.py

Typed exception hierarchy for the entire application.

All domain exceptions inherit from AppError, which carries:
  - message       — human-readable description
  - code          — machine-readable slug (for API consumers)
  - status_code   — HTTP status to return
  - detail        — optional structured payload (passed through to the response)

The global exception handler in gateway/middleware/error_handler.py maps
AppError subclasses to uniform JSON responses.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all application errors."""

    message: str = "An unexpected error occurred."
    code: str = "internal_error"
    status_code: int = 500

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        detail: Any = None,
        status_code: int | None = None,
    ) -> None:
        self.message = message or self.__class__.message
        self.code = code or self.__class__.code
        self.detail = detail
        self.status_code = status_code or self.__class__.status_code
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": self.code, "message": self.message}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload



# 4xx — client errors

class NotFoundError(AppError):
    """Resource not found."""
    message = "Resource not found."
    code = "not_found"
    status_code = 404


class ValidationError(AppError):
    """Request payload failed validation."""
    message = "Validation failed."
    code = "validation_error"
    status_code = 422


class AuthenticationError(AppError):
    """Missing or invalid credentials."""
    message = "Authentication required."
    code = "authentication_error"
    status_code = 401


class AuthorizationError(AppError):
    """Authenticated but not authorised for this action."""
    message = "You do not have permission to perform this action."
    code = "authorization_error"
    status_code = 403


class RateLimitError(AppError):
    """Too many requests."""
    message = "Rate limit exceeded. Please slow down."
    code = "rate_limit_exceeded"
    status_code = 429


class ConflictError(AppError):
    """Resource already exists or state conflict."""
    message = "Conflict with existing resource."
    code = "conflict"
    status_code = 409



# 5xx — server / infrastructure errors

class LLMError(AppError):
    """LLM provider returned an error or timed out."""
    message = "LLM provider error."
    code = "llm_error"
    status_code = 502


class LLMTimeoutError(LLMError):
    """LLM provider did not respond in time."""
    message = "LLM provider timed out."
    code = "llm_timeout"
    status_code = 504


class EmbeddingError(AppError):
    """Embedding service error."""
    message = "Embedding service error."
    code = "embedding_error"
    status_code = 502


class RerankerError(AppError):
    """Reranker service error."""
    message = "Reranker service error."
    code = "reranker_error"
    status_code = 502


class StorageError(AppError):
    """Database or vector store error."""
    message = "Storage backend error."
    code = "storage_error"
    status_code = 503


class CacheError(AppError):
    """Redis or cache layer error."""
    message = "Cache layer error."
    code = "cache_error"
    status_code = 503


class ToolError(AppError):
    """Tool execution failed."""
    message = "Tool execution failed."
    code = "tool_error"
    status_code = 500

    def __init__(
        self,
        message: str | None = None,
        *,
        tool_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.tool_name = tool_name


class SandboxError(AppError):
    """Code execution sandbox error (timeout, OOM, escape attempt)."""
    message = "Sandbox execution error."
    code = "sandbox_error"
    status_code = 500


class MemoryError(AppError):
    """Memory service error."""
    message = "Memory service error."
    code = "memory_error"
    status_code = 500


class AgentError(AppError):
    """Agent graph execution error."""
    message = "Agent execution error."
    code = "agent_error"
    status_code = 500


class BudgetExceededError(AgentError):
    """Agent exceeded its token/iteration/time budget."""
    message = "Agent budget exceeded."
    code = "budget_exceeded"
    status_code = 500


class PolicyViolationError(AppError):
    """Request violates the active policy."""
    message = "Request violates active policy."
    code = "policy_violation"
    status_code = 403


class ConfigurationError(AppError):
    """Misconfiguration detected at startup."""
    message = "Server misconfiguration."
    code = "configuration_error"
    status_code = 500


class ExternalServiceError(AppError):
    """Upstream external service (web search, MCP, etc.) error."""
    message = "External service error."
    code = "external_service_error"
    status_code = 502
