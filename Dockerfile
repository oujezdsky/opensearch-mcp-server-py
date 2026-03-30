FROM python:3.10-slim AS builder

# Install system deps
RUN apt-get update && apt-get install -y gcc libpq-dev curl && rm -rf /var/lib/apt/lists/*

# Install uv via pip
RUN pip install uv

# Create venv
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# Copy all necessary files for building
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY tests/ ./tests/

# Sync all dependencies (main + dev)
RUN uv sync --locked --all-extras

# Final stage
FROM python:3.10-slim

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv globally
RUN pip install uv

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy project files (potřebné pro instalaci)
COPY pyproject.toml uv.lock config.yml test_config_infrastructure.py agentic_test_routine.py ./
COPY src/ ./src/
COPY tests/ ./tests/

# Instaluj projekt v editable režimu
RUN uv pip install -e .

# Command (bude přepsán compose)
CMD ["uv", "run", "pytest", "-v"]