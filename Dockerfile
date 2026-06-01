FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ChromaDB's ONNX embedding model so it's baked into the image
RUN python -c "import chromadb; c = chromadb.EphemeralClient(); c.get_or_create_collection('warmup'); c.delete_collection('warmup')"

COPY . .

EXPOSE 8000

ENV TOKENIZERS_PARALLELISM=false
ENV ANONYMIZED_TELEMETRY=false

CMD uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}
