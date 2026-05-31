"""AST-aware code chunker that preserves semantic boundaries.

Unlike naive text splitting, this understands code structure:
- Functions and classes become individual chunks
- Imports and module-level code are grouped
- Docstrings and comments stay attached to their code
- Each chunk carries metadata (file path, line numbers, symbol names)
"""

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodeChunk:
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # "function", "class", "module_level", "import"
    symbols: list[str] = field(default_factory=list)
    language: str = "python"

    @property
    def metadata(self) -> dict:
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "symbols": self.symbols,
            "language": self.language,
        }


class PythonChunker:
    """Chunks Python files using AST parsing for semantic boundaries."""

    def __init__(self, max_chunk_size: int = 1500, overlap_lines: int = 2):
        self.max_chunk_size = max_chunk_size
        self.overlap_lines = overlap_lines

    def chunk_file(self, file_path: str, content: str) -> list[CodeChunk]:
        try:
            tree = ast.parse(content)
            return self._chunk_from_ast(tree, file_path, content)
        except SyntaxError:
            return self._chunk_by_lines(file_path, content)

    def _chunk_from_ast(
        self, tree: ast.Module, file_path: str, source: str
    ) -> list[CodeChunk]:
        lines = source.splitlines(keepends=True)
        chunks = []

        imports = []
        module_level = []
        functions_and_classes = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions_and_classes.append(("function", node))
            elif isinstance(node, ast.ClassDef):
                functions_and_classes.append(("class", node))
            else:
                module_level.append(node)

        if imports:
            start = imports[0].lineno
            end = imports[-1].end_lineno or imports[-1].lineno
            chunk_content = "".join(lines[start - 1 : end])
            chunks.append(
                CodeChunk(
                    content=chunk_content.strip(),
                    file_path=file_path,
                    start_line=start,
                    end_line=end,
                    chunk_type="import",
                    symbols=[self._import_name(n) for n in imports],
                )
            )

        for kind, node in functions_and_classes:
            start = node.lineno
            end = node.end_lineno or node.lineno
            chunk_content = "".join(lines[start - 1 : end])

            symbols = [node.name]
            if kind == "class":
                for item in ast.walk(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append(f"{node.name}.{item.name}")

            chunk = CodeChunk(
                content=chunk_content.strip(),
                file_path=file_path,
                start_line=start,
                end_line=end,
                chunk_type=kind,
                symbols=symbols,
            )

            if len(chunk.content) > self.max_chunk_size and kind == "class":
                chunks.extend(self._split_class(node, file_path, lines))
            else:
                chunks.append(chunk)

        if module_level:
            start = module_level[0].lineno
            end = module_level[-1].end_lineno or module_level[-1].lineno
            chunk_content = "".join(lines[start - 1 : end])
            if chunk_content.strip():
                chunks.append(
                    CodeChunk(
                        content=chunk_content.strip(),
                        file_path=file_path,
                        start_line=start,
                        end_line=end,
                        chunk_type="module_level",
                    )
                )

        return chunks if chunks else self._chunk_by_lines(file_path, source)

    def _split_class(
        self, class_node: ast.ClassDef, file_path: str, lines: list[str]
    ) -> list[CodeChunk]:
        chunks = []
        for node in ast.iter_child_nodes(class_node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno
                end = node.end_lineno or node.lineno
                chunk_content = "".join(lines[start - 1 : end])
                chunks.append(
                    CodeChunk(
                        content=chunk_content.strip(),
                        file_path=file_path,
                        start_line=start,
                        end_line=end,
                        chunk_type="method",
                        symbols=[f"{class_node.name}.{node.name}"],
                    )
                )
        return chunks

    def _chunk_by_lines(self, file_path: str, content: str) -> list[CodeChunk]:
        lines = content.splitlines()
        chunks = []
        chunk_size = 40

        for i in range(0, len(lines), chunk_size - self.overlap_lines):
            chunk_lines = lines[i : i + chunk_size]
            if not "".join(chunk_lines).strip():
                continue
            chunks.append(
                CodeChunk(
                    content="\n".join(chunk_lines),
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=min(i + chunk_size, len(lines)),
                    chunk_type="raw",
                )
            )

        return chunks

    def _import_name(self, node: ast.Import | ast.ImportFrom) -> str:
        if isinstance(node, ast.ImportFrom):
            return node.module or ""
        return node.names[0].name if node.names else ""


class GenericChunker:
    """Line-based chunker for non-Python files (JS, TS, Go, Rust, etc.)."""

    SUPPORTED_EXTENSIONS = {
        ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
        ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs",
        ".swift", ".kt", ".scala", ".sh", ".yaml", ".yml",
        ".toml", ".json", ".md", ".txt", ".sql", ".html", ".css",
    }

    def __init__(self, chunk_size: int = 60, overlap: int = 5):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_file(self, file_path: str, content: str) -> list[CodeChunk]:
        ext = Path(file_path).suffix.lower()
        language = self._detect_language(ext)
        lines = content.splitlines()
        chunks = []

        for i in range(0, len(lines), self.chunk_size - self.overlap):
            chunk_lines = lines[i : i + self.chunk_size]
            chunk_content = "\n".join(chunk_lines)
            if not chunk_content.strip():
                continue
            chunks.append(
                CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=min(i + self.chunk_size, len(lines)),
                    chunk_type="raw",
                    language=language,
                )
            )

        return chunks

    def _detect_language(self, ext: str) -> str:
        mapping = {
            ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
            ".jsx": "javascript", ".go": "go", ".rs": "rust",
            ".java": "java", ".rb": "ruby", ".php": "php",
            ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
            ".cs": "csharp", ".swift": "swift", ".kt": "kotlin",
            ".scala": "scala", ".sh": "shell", ".sql": "sql",
            ".md": "markdown", ".html": "html", ".css": "css",
            ".yaml": "yaml", ".yml": "yaml", ".json": "json",
            ".toml": "toml",
        }
        return mapping.get(ext, "text")


def chunk_repository(repo_path: str) -> list[CodeChunk]:
    """Walk a repository and chunk all supported files."""
    python_chunker = PythonChunker()
    generic_chunker = GenericChunker()
    all_chunks = []

    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".tox", ".mypy_cache"}

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for file_name in files:
            file_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(file_path, repo_path)
            ext = Path(file_name).suffix.lower()

            if ext == ".py":
                try:
                    content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
                    chunks = python_chunker.chunk_file(rel_path, content)
                    all_chunks.extend(chunks)
                except Exception:
                    continue
            elif ext in GenericChunker.SUPPORTED_EXTENSIONS:
                try:
                    content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
                    if len(content) > 100_000:
                        continue
                    chunks = generic_chunker.chunk_file(rel_path, content)
                    all_chunks.extend(chunks)
                except Exception:
                    continue

    return all_chunks
