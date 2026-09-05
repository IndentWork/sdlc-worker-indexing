FROM python:3.13-slim

WORKDIR /app

# Install uv for fast, reproducible dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files — Docker layer cache: only reinstalls when these change
COPY pyproject.toml uv.lock* ./

# Install dependencies exactly as pinned in the lock file
RUN uv sync --frozen --no-dev

# Copy application code
COPY app/ ./app/

CMD ["uv", "run", "python", "-m", "app.main"]
