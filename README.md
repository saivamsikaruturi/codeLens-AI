# CodeRAG — Intelligent Codebase Q&A

> Point at any GitHub repo. Ask questions in plain English. Get accurate, cited answers.

CodeRAG uses Retrieval-Augmented Generation (RAG) with **AST-aware code chunking** to provide intelligent answers about any codebase. Unlike naive text-splitting approaches, it understands code structure — keeping functions, classes, and imports as semantic units.

## Architecture

```
GitHub Repo → Clone → AST Parse → Chunk → Embed → ChromaDB
                                                      ↓
User Query → Embed → Vector Search → Top-K Chunks → Claude API → Cited Answer
```

### Key Design Decisions

| Decision | Why |
|----------|-----|
| AST-aware chunking | Preserves semantic boundaries (functions, classes) instead of splitting mid-logic |
| Metadata-rich indexing | Enables filtering by file path, symbol name, chunk type |
| ChromaDB (local) | Zero-config vector store, no external dependencies for dev |
| Claude API | 200K context window handles large code contexts; strong code reasoning |
| FastAPI | Async, auto-documented API with Pydantic validation |

## Features

- **AST-Aware Chunking** — Python files are parsed into functions, classes, and imports. Other languages use intelligent line-based splitting.
- **Hybrid Metadata** — Each chunk carries file path, line numbers, language, and symbol names for precise retrieval.
- **Source Citations** — Every answer includes exact file paths and line numbers.
- **Multi-Language** — Supports Python, JavaScript, TypeScript, Go, Rust, Java, and 15+ more languages.
- **Clean API** — RESTful endpoints with OpenAPI docs at `/docs`.
- **Modern UI** — Dark-themed chat interface with real-time indexing status.

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/coderag.git
cd coderag

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure
export ANTHROPIC_API_KEY=your-key-here

# Run
uvicorn backend.app:app --reload --port 8000
```

Open http://localhost:8000 — paste a GitHub URL, click Index, then ask questions.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ingest` | Clone and index a GitHub repository |
| POST | `/api/query` | Ask a question about indexed code |
| GET | `/api/repos` | List all indexed repositories |
| DELETE | `/api/repos/{name}` | Remove an indexed repository |

## Example Queries

After indexing a repo, try:
- "How does the authentication middleware work?"
- "What design patterns are used in this project?"
- "Explain the data flow from API request to database"
- "Where are environment variables configured?"
- "What are the main dependencies and why?"

## Running Tests

```bash
pip install -e ".[dev]"
pytest -v
```

## Docker

```bash
docker build -t coderag .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your-key coderag
```

## Tech Stack

- **Backend**: Python 3.12, FastAPI, Pydantic
- **LLM**: Claude API (Anthropic)
- **Vector Store**: ChromaDB (embedded, persistent)
- **Chunking**: Python AST + language-aware line splitting
- **Frontend**: Vanilla HTML/CSS/JS (no build step)
- **Deployment**: Docker, single container

## Project Structure

```
coderag/
├── backend/
│   ├── app.py              # FastAPI application & routes
│   ├── ingestion/
│   │   ├── chunker.py      # AST-aware code chunking engine
│   │   └── github_loader.py # Git clone & repo management
│   ├── retrieval/
│   │   └── vector_store.py # ChromaDB vector search
│   ├── generation/
│   │   └── llm.py          # Claude API integration
│   └── models/
│       └── schemas.py      # Pydantic request/response models
├── frontend/
│   └── index.html          # Chat UI
├── tests/
│   ├── test_chunker.py     # Chunker unit tests
│   └── test_api.py         # API integration tests
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

## License

MIT
