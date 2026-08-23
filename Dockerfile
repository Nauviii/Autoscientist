# Execution is pinned to Linux so the environment fingerprint means something.
# The sandbox needs fork and setrlimit, neither of which exists on Windows, and a
# score produced under one BLAS build must reproduce under the same one.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# libgomp1 is required by LightGBM and XGBoost; the slim image omits it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependencies are installed before the source is copied so a code edit does not
# invalidate the layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra dev

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
RUN uv sync --frozen --extra dev

CMD ["uv", "run", "pytest", "-q"]
