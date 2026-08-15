FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

# Compile hnswlib from source (avoids pre-built AVX2/AVX-512 wheel).
# duckdb pinned to 1.0.0 — 1.5.x uses AVX-512 which Render free-tier CPUs lack.
RUN pip install --no-cache-dir --no-binary hnswlib -r requirements.txt

COPY . .

RUN mkdir -p data/hnsw_indices && chmod -R 777 data/hnsw_indices

EXPOSE 10000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
