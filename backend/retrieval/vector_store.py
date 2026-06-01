"""Vector store using ChromaDB with Gemini API embeddings (no local ONNX model)."""

import hashlib
import json
import os
import urllib.request
import urllib.error
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb import Documents, EmbeddingFunction, Embeddings

from backend.ingestion.chunker import CodeChunk


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Compute embeddings via Gemini API instead of loading a local ONNX model."""

    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        self.api_key = api_key
        self.model = model
        self.url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
            f"?key={api_key}"
        )

    def __call__(self, input: Documents) -> Embeddings:
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(input), batch_size):
            batch = input[i : i + batch_size]
            payload = {
                "requests": [
                    {"model": f"models/{self.model}", "content": {"parts": [{"text": text}]}}
                    for text in batch
                ]
            }

            req = urllib.request.Request(
                self.url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            for embedding in result["embeddings"]:
                all_embeddings.append(embedding["values"])

        return all_embeddings


class VectorStore:
    """ChromaDB-backed vector store with Gemini API embeddings."""

    def __init__(self, persist_dir: str = "./chroma_data"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        api_key = os.environ.get("GEMINI_API_KEY", "")
        self.embedding_fn = GeminiEmbeddingFunction(api_key=api_key) if api_key else None

    def get_or_create_collection(self, repo_name: str) -> chromadb.Collection:
        safe_name = repo_name.replace("-", "_").replace(".", "_")[:60]
        kwargs = {"name": safe_name, "metadata": {"hnsw:space": "cosine"}}
        if self.embedding_fn:
            kwargs["embedding_function"] = self.embedding_fn
        return self.client.get_or_create_collection(**kwargs)

    def index_chunks(self, repo_name: str, chunks: list[CodeChunk]) -> int:
        collection = self.get_or_create_collection(repo_name)

        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            doc_text = self._format_chunk_for_embedding(chunk)
            doc_id = hashlib.md5(
                f"{chunk.file_path}:{chunk.start_line}:{chunk.content[:100]}".encode()
            ).hexdigest()

            documents.append(doc_text)
            metadatas.append({
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_type": chunk.chunk_type,
                "symbols": ",".join(chunk.symbols),
                "language": chunk.language,
            })
            ids.append(doc_id)

        batch_size = 100
        for i in range(0, len(documents), batch_size):
            collection.upsert(
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
                ids=ids[i : i + batch_size],
            )

        return len(documents)

    def search(
        self,
        repo_name: str,
        query: str,
        top_k: int = 5,
        file_filter: Optional[str] = None,
        chunk_type_filter: Optional[str] = None,
    ) -> list[dict]:
        collection = self.get_or_create_collection(repo_name)

        where_filter = None
        conditions = []
        if file_filter:
            conditions.append({"file_path": {"$contains": file_filter}})
        if chunk_type_filter:
            conditions.append({"chunk_type": chunk_type_filter})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                relevance = 1 - distance
                search_results.append({
                    "content": doc,
                    "metadata": meta,
                    "relevance_score": round(max(0, relevance), 4),
                })

        return search_results

    def delete_collection(self, repo_name: str):
        safe_name = repo_name.replace("-", "_").replace(".", "_")[:60]
        try:
            self.client.delete_collection(safe_name)
        except ValueError:
            pass

    def list_collections(self) -> list[str]:
        return [col.name for col in self.client.list_collections()]

    def _format_chunk_for_embedding(self, chunk: CodeChunk) -> str:
        header = f"File: {chunk.file_path} | Type: {chunk.chunk_type}"
        if chunk.symbols:
            header += f" | Symbols: {', '.join(chunk.symbols[:5])}"
        return f"{header}\n\n{chunk.content}"
