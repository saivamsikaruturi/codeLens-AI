"""FastAPI application for CodeLens AI — Intelligent Codebase Q&A powered by RAG."""

import os
import time
import json
from contextlib import asynccontextmanager
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from backend.ingestion.chunker import chunk_repository
from backend.ingestion.github_loader import GitHubLoader
from backend.generation.llm import CodeLensGenerator
from backend.models.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.reranker import GeminiReranker
from backend.retrieval.file_tree import build_file_tree


@asynccontextmanager
async def lifespan(app: FastAPI):
    chroma_dir = os.environ.get("CHROMA_DATA_DIR", "./chroma_data")
    app.state.vector_store = VectorStore(persist_dir=chroma_dir)
    app.state.github_loader = GitHubLoader()
    app.state.query_history = []
    app.state.metrics = defaultdict(lambda: {
        "total_queries": 0,
        "avg_confidence": 0.0,
        "avg_latency_ms": 0,
        "total_chunks_retrieved": 0,
    })

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        app.state.generator = CodeLensGenerator(api_key=api_key)
        app.state.reranker = GeminiReranker(api_key=api_key)
    else:
        app.state.generator = None
        app.state.reranker = None

    yield


app = FastAPI(
    title="CodeLens AI",
    description="Intelligent Codebase Q&A powered by RAG with AST-aware chunking",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_repository(request: IngestRequest):
    """Clone and index a GitHub repository."""
    try:
        loader: GitHubLoader = app.state.github_loader
        local_path, repo_name = loader.load_repo(request.repo_url, request.branch)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    chunks = chunk_repository(local_path)
    if not chunks:
        raise HTTPException(status_code=400, detail="No indexable files found in repository")

    vector_store: VectorStore = app.state.vector_store
    num_indexed = vector_store.index_chunks(repo_name, chunks)

    loader.cleanup(repo_name)

    return IngestResponse(
        repo_name=repo_name,
        files_indexed=len(set(c.file_path for c in chunks)),
        chunks_created=num_indexed,
        status="success",
    )


@app.post("/api/query", response_model=QueryResponse)
async def query_codebase(request: QueryRequest):
    """Ask a question about an indexed codebase."""
    generator: CodeLensGenerator | None = app.state.generator
    if not generator:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not configured. Set it as an environment variable.",
        )

    start_time = time.time()

    vector_store: VectorStore = app.state.vector_store
    retrieved = vector_store.search(
        repo_name=request.repo_name,
        query=request.question,
        top_k=request.top_k * 2,
    )

    if not retrieved:
        raise HTTPException(
            status_code=404,
            detail=f"No indexed data found for '{request.repo_name}'. Ingest the repository first.",
        )

    # Rerank for better precision
    reranker = app.state.reranker
    if reranker and len(retrieved) > request.top_k:
        retrieved = reranker.rerank(request.question, retrieved, top_k=request.top_k)
    else:
        retrieved = retrieved[:request.top_k]

    try:
        result = generator.generate_answer(
            question=request.question,
            retrieved_chunks=retrieved,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

    latency_ms = int((time.time() - start_time) * 1000)

    # Record metrics
    metrics = app.state.metrics[request.repo_name]
    metrics["total_queries"] += 1
    n = metrics["total_queries"]
    metrics["avg_confidence"] = round(
        ((metrics["avg_confidence"] * (n - 1)) + result["confidence"]) / n, 3
    )
    metrics["avg_latency_ms"] = int(
        ((metrics["avg_latency_ms"] * (n - 1)) + latency_ms) / n
    )
    metrics["total_chunks_retrieved"] += len(retrieved)

    # Record history
    app.state.query_history.append({
        "question": request.question,
        "repo_name": request.repo_name,
        "confidence": result["confidence"],
        "latency_ms": latency_ms,
        "sources_count": len(result["sources"]),
        "timestamp": time.time(),
    })
    if len(app.state.query_history) > 100:
        app.state.query_history = app.state.query_history[-100:]

    return QueryResponse(
        answer=result["answer"],
        sources=[
            SourceChunk(
                file_path=s["file_path"],
                start_line=s["start_line"],
                end_line=s["end_line"],
                content=s["content"],
                relevance_score=s["relevance_score"],
            )
            for s in result["sources"]
        ],
        confidence=result["confidence"],
    )


@app.post("/api/query/stream")
async def query_codebase_stream(request: QueryRequest):
    """Stream answer tokens via Server-Sent Events."""
    generator: CodeLensGenerator | None = app.state.generator
    if not generator:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not configured.",
        )

    vector_store: VectorStore = app.state.vector_store
    retrieved = vector_store.search(
        repo_name=request.repo_name,
        query=request.question,
        top_k=request.top_k * 2,
    )

    if not retrieved:
        raise HTTPException(
            status_code=404,
            detail=f"No indexed data found for '{request.repo_name}'.",
        )

    reranker = app.state.reranker
    if reranker and len(retrieved) > request.top_k:
        retrieved = reranker.rerank(request.question, retrieved, top_k=request.top_k)
    else:
        retrieved = retrieved[:request.top_k]

    sources = [
        {
            "file_path": chunk["metadata"]["file_path"],
            "start_line": chunk["metadata"]["start_line"],
            "end_line": chunk["metadata"]["end_line"],
            "relevance_score": chunk["relevance_score"],
        }
        for chunk in retrieved
    ]

    def event_stream():
        # Send sources first
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        try:
            for token in generator.generate_stream(
                question=request.question,
                retrieved_chunks=retrieved,
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except RuntimeError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/repos")
async def list_repos():
    """List all indexed repositories."""
    vector_store: VectorStore = app.state.vector_store
    return {"repositories": vector_store.list_collections()}


@app.delete("/api/repos/{repo_name}")
async def delete_repo(repo_name: str):
    """Delete an indexed repository."""
    vector_store: VectorStore = app.state.vector_store
    vector_store.delete_collection(repo_name)
    return {"status": "deleted", "repo_name": repo_name}


@app.get("/api/repos/{repo_name}/tree")
async def get_file_tree(repo_name: str):
    """Get file tree structure of an indexed repository."""
    vector_store: VectorStore = app.state.vector_store
    collection = vector_store.get_or_create_collection(repo_name)

    results = collection.get(include=["metadatas"])
    if not results["metadatas"]:
        raise HTTPException(status_code=404, detail="Repository not found")

    tree = build_file_tree(results["metadatas"])
    return {"repo_name": repo_name, "tree": tree}


@app.get("/api/history")
async def get_query_history(limit: int = 20):
    """Get recent query history."""
    history = app.state.query_history[-limit:]
    history.reverse()
    return {"history": history, "total": len(app.state.query_history)}


@app.delete("/api/history")
async def clear_history():
    """Clear query history."""
    app.state.query_history = []
    return {"status": "cleared"}


@app.get("/api/metrics")
async def get_metrics():
    """Get evaluation metrics dashboard data."""
    metrics = dict(app.state.metrics)
    total_queries = sum(m["total_queries"] for m in metrics.values())
    avg_confidence = (
        sum(m["avg_confidence"] * m["total_queries"] for m in metrics.values()) / total_queries
        if total_queries > 0
        else 0
    )
    avg_latency = (
        sum(m["avg_latency_ms"] * m["total_queries"] for m in metrics.values()) / total_queries
        if total_queries > 0
        else 0
    )

    return {
        "global": {
            "total_queries": total_queries,
            "avg_confidence": round(avg_confidence, 3),
            "avg_latency_ms": int(avg_latency),
            "repos_indexed": len(metrics),
        },
        "per_repo": metrics,
    }


# Serve frontend
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
frontend_dir = os.path.normpath(frontend_dir)


@app.get("/")
async def serve_frontend():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "CodeLens AI API is running", "docs": "/docs"}


@app.head("/")
async def health_check():
    from fastapi.responses import Response
    return Response(status_code=200)
