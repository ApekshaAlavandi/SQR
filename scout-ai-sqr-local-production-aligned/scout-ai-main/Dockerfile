FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install .

# Run as a non-root user.
RUN useradd --create-home --uid 10001 scout
USER scout


CMD ["scout-ai-consumer"]
