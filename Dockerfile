FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

# Install hnswlib from source so it compiles for this machine's CPU
# rather than using the PyPI binary wheel (which assumes AVX2/AVX-512)
RUN pip install --no-cache-dir --no-binary hnswlib -r requirements.txt

COPY . .

RUN mkdir -p data/hnsw_indices && chmod -R 777 data/hnsw_indices

EXPOSE 10000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
