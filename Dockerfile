FROM python:3.12-slim

WORKDIR /app

ENV TOKENIZERS_PARALLELISM=false
ENV ANONYMIZED_TELEMETRY=false

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y onnxruntime tokenizers torch && \
    rm -rf /root/.cache

COPY . .

EXPOSE 8000

CMD uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}
