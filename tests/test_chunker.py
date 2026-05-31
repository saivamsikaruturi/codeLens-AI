"""Tests for the AST-aware code chunker."""

from backend.ingestion.chunker import PythonChunker, GenericChunker, CodeChunk


SAMPLE_PYTHON = '''
import os
from pathlib import Path

DB_URL = "postgres://localhost/mydb"

def connect_database(url: str) -> None:
    """Connect to the database."""
    print(f"Connecting to {url}")

class UserService:
    """Handles user operations."""

    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int):
        return self.db.query(user_id)

    def create_user(self, name: str, email: str):
        return self.db.insert({"name": name, "email": email})
'''


def test_python_chunker_extracts_functions():
    chunker = PythonChunker()
    chunks = chunker.chunk_file("app/service.py", SAMPLE_PYTHON)

    function_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(function_chunks) == 1
    assert "connect_database" in function_chunks[0].symbols


def test_python_chunker_extracts_classes():
    chunker = PythonChunker()
    chunks = chunker.chunk_file("app/service.py", SAMPLE_PYTHON)

    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) == 1
    assert "UserService" in class_chunks[0].symbols
    assert "UserService.get_user" in class_chunks[0].symbols


def test_python_chunker_extracts_imports():
    chunker = PythonChunker()
    chunks = chunker.chunk_file("app/service.py", SAMPLE_PYTHON)

    import_chunks = [c for c in chunks if c.chunk_type == "import"]
    assert len(import_chunks) == 1
    assert import_chunks[0].start_line == 2


def test_python_chunker_preserves_line_numbers():
    chunker = PythonChunker()
    chunks = chunker.chunk_file("app/service.py", SAMPLE_PYTHON)

    for chunk in chunks:
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line
        assert chunk.file_path == "app/service.py"


def test_python_chunker_handles_syntax_error():
    chunker = PythonChunker()
    bad_code = "def broken(:\n    pass"
    chunks = chunker.chunk_file("broken.py", bad_code)
    assert len(chunks) > 0
    assert chunks[0].chunk_type == "raw"


def test_generic_chunker_handles_javascript():
    chunker = GenericChunker(chunk_size=10, overlap=2)
    js_code = "\n".join([f"const line{i} = {i};" for i in range(25)])
    chunks = chunker.chunk_file("app.js", js_code)

    assert len(chunks) > 1
    assert chunks[0].language == "javascript"


def test_chunk_metadata():
    chunker = PythonChunker()
    chunks = chunker.chunk_file("test.py", SAMPLE_PYTHON)

    for chunk in chunks:
        meta = chunk.metadata
        assert "file_path" in meta
        assert "start_line" in meta
        assert "chunk_type" in meta
        assert meta["language"] == "python"
