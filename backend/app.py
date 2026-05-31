"""FastAPI application for CodeRAG — Codebase Q&A powered by RAG."""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.ingestion.chunker import chunk_repository
from backend.ingestion.github_loader import GitHubLoader
from backend.generation.llm import CodeRAGGenerator
from backend.models.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from backend.retrieval.vector_store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vector_store = VectorStore(persist_dir="./chroma_data")
    app.state.github_loader = GitHubLoader()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        app.state.generator = CodeRAGGenerator(api_key=api_key)
    else:
        app.state.generator = None

    yield


app = FastAPI(
    title="CodeRAG",
    description="Intelligent Codebase Q&A powered by RAG with AST-aware chunking",
    version="1.0.0",
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
    generator: CodeRAGGenerator | None = app.state.generator
    if not generator:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not configured. Set it as an environment variable.",
        )

    vector_store: VectorStore = app.state.vector_store
    retrieved = vector_store.search(
        repo_name=request.repo_name,
        query=request.question,
        top_k=request.top_k,
    )

    if not retrieved:
        raise HTTPException(
            status_code=404,
            detail=f"No indexed data found for '{request.repo_name}'. Ingest the repository first.",
        )

    result = generator.generate_answer(
        question=request.question,
        retrieved_chunks=retrieved,
    )

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


# Serve frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="frontend")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
