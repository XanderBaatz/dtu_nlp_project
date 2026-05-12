ARG UV_VERSION=latest
ARG PYTHON_VERSION=3.13


# ── Stage 1: export a minimal dependency set ─────────────────────────────────
FROM ghcr.io/astral-sh/uv:$UV_VERSION AS uv


# ── Stage 2: build venv with all runtime deps ────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (better layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Download spaCy model into the venv (invoke venv python directly to avoid
# uv trying to build the project editable and requiring LICENSE etc.)
RUN /app/.venv/bin/python -m spacy download en_core_web_sm


# ── Stage 3: lean runtime image ───────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy the pre-built venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/movies ./src/movies
COPY src/__init__.py ./src/__init__.py
COPY tools ./tools

# Data directories are expected to be mounted at runtime:
#   docker run -v /host/data:/app/data ...
# If you want to bake data into the image instead, uncomment:
#   COPY data/ ./data/

EXPOSE 8002

CMD ["uvicorn", "movies.main:app", "--host", "0.0.0.0", "--port", "8002"]
