FROM python:3.12-slim

WORKDIR /app

ENV TOKENIZERS_PARALLELISM=false
ENV ANONYMIZED_TELEMETRY=false
# Pin the model cache to a known location that persists across user contexts
ENV XDG_CACHE_HOME=/app/.cache

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ChromaDB's ONNX embedding model by triggering an actual embedding
RUN python -c "\
import chromadb;\
c = chromadb.EphemeralClient();\
col = c.get_or_create_collection('warmup');\
col.add(documents=['hello world'], ids=['1']);\
c.delete_collection('warmup')\
" && ls -la /app/.cache/chroma/onnx_models/all-MiniLM-L6-v2/

COPY . .

EXPOSE 8000

CMD uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}
