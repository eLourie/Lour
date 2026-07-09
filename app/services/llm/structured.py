"""
app/services/llm/structured.py

Structured output wrapper: LLM → validated Pydantic model.

Uses Instructor under the hood. On ValidationError the LLM receives
feedback about what was wrong and retries (up to MAX_RETRIES times).
This is the "retry-with-feedback" pattern referenced in ADR-007.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.exceptions import LLMError
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.services.llm.base import LLMMessage, LLMProvider

logger = get_logger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

MAX_RETRIES = 2

_SYSTEM_PROMPT = (
    "You are a precise assistant. Always respond with a single valid JSON object "
    "that matches the schema provided. No markdown, no explanation — only JSON."
)


class StructuredOutputService:
    """
    Wraps any LLMProvider and adds schema-enforced structured output.

    Usage::

        class Route(BaseModel):
            agent: str
            reasoning: str

        svc = StructuredOutputService(provider)
        result: Route = await svc.complete(messages, schema=Route)
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def complete(
        self,
        messages: list[LLMMessage],
        schema: type[ModelT],
        *,
        context: str = "",
    ) -> ModelT:
        """
        Ask the LLM to produce an instance of *schema*.

        The schema's JSON schema is injected as a system message.
        On ValidationError, the error is fed back to the LLM for
        one more attempt before giving up.
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        system_msg: LLMMessage = {
            "role": "system",
            "content": (
                f"{_SYSTEM_PROMPT}\n\nExpected JSON schema:\n```json\n{schema_json}\n```"
                + (f"\n\nContext: {context}" if context else "")
            ),
        }
        full_messages = [system_msg, *messages]

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 2):  # +2: initial + MAX_RETRIES
            response = await self._provider.chat(full_messages)
            raw = response.content.strip()

            # Strip markdown fences if the model wrapped the JSON
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

            try:
                data: Any = json.loads(raw)
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                if attempt <= MAX_RETRIES:
                    logger.warning(
                        "structured_output_retry",
                        attempt=attempt,
                        error=str(exc),
                        schema=schema.__name__,
                    )
                    feedback_msg: LLMMessage = {
                        "role": "user",
                        "content": (
                            f"Your previous response was invalid.\n"
                            f"Error: {exc}\n"
                            f"Raw response: {raw}\n\n"
                            "Please respond again with a valid JSON object matching the schema."
                        ),
                    }
                    full_messages = [
                        *full_messages,
                        {"role": "assistant", "content": raw},
                        feedback_msg,
                    ]

        raise LLMError(
            f"Structured output failed after {MAX_RETRIES + 1} attempts "
            f"for schema {schema.__name__}: {last_error}"
        )
