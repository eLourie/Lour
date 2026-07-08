# =============================================================================
#  AI Agent — Multi-stage Dockerfile
#  Stage 1 (builder): installs deps with uv into /app/.venv
#  Stage 2 (runtime): copies venv, runs as non-root user
# =============================================================================


# Stage 1 — builder

FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first (layer-cache friendly)
COPY pyproject.toml ./
# uv.lock is committed to the repo — copy it if present
COPY uv.lock* ./

# Install production dependencies only (no dev extras)
RUN uv sync --no-dev --frozen

# Copy application source
COPY app/ ./app/
COPY migrations/ ./migrations/
COPY scripts/ ./scripts/
COPY alembic.ini* ./


# Stage 2 — runtime

FROM python:3.12-slim AS runtime

# Security: run as non-root
RUN groupadd --gid 1001 appuser \
 && useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed venv from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app /app/app
COPY --from=builder /app/migrations /app/migrations
COPY --from=builder /app/scripts /app/scripts
COPY --from=builder /app/alembic.ini* /app/

# Activate venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]