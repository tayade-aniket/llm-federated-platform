FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_TF=0 \
    USE_JAX=0 \
    USE_TORCH=1 \
    HF_HOME=/tmp/huggingface_cache

WORKDIR /app

# Install system dependencies (needed for compiling certain ML modules if required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements-render.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-render.txt

# Copy application files
COPY . .

# Ensure standard directories exist and are writable
RUN mkdir -p data adapters models/cache && \
    chmod -R 777 data adapters models

# Expose ports for Web UI (8000) and Flower Federated Server (8080)
EXPOSE 8000 8080

# Start the platform orchestration script by default (can be overridden)
CMD ["python", "run_platform.py", "--mode", "all"]
