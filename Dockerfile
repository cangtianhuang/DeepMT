# DeepMT Docker Image
#
# GPU build (default):
#   docker build -t deepmt .
#   docker run --gpus all --rm -e OPENAI_API_KEY=sk-... deepmt deepmt health check
#
# CPU-only build (for machines without NVIDIA GPU):
#   docker build --build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cpu -t deepmt-cpu .
#   docker run --rm deepmt-cpu deepmt health check
#
# Start Web Dashboard:
#   docker run --gpus all -p 8000:8000 -e OPENAI_API_KEY=sk-... deepmt deepmt ui start --host 0.0.0.0

ARG PYTORCH_INDEX=https://download.pytorch.org/whl/cu121
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (layer caching)
COPY pyproject.toml requirements.txt ./

# Install PyTorch — GPU by default, override PYTORCH_INDEX for CPU-only
ARG PYTORCH_INDEX=https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir torch --index-url ${PYTORCH_INDEX}

# Copy source and install all extras (llm, web-search, sympy, ui)
COPY . .
RUN pip install --no-cache-dir -e ".[all]"

# Create data directories
RUN mkdir -p data/logs data/knowledge data/results data/cache

# Default config from example (user can mount their own config.yaml)
RUN cp config.yaml.example config.yaml

ENV PYTHONPATH=/app
ENV DEEPMT_LOG_CONSOLE_STYLE=file

EXPOSE 8000

CMD ["deepmt", "--help"]
