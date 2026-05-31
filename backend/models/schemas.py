from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    repo_url: str = Field(description="GitHub repository URL to ingest")
    branch: str = Field(default="main", description="Branch to index")


class IngestResponse(BaseModel):
    repo_name: str
    files_indexed: int
    chunks_created: int
    status: str


class QueryRequest(BaseModel):
    question: str = Field(description="Natural language question about the codebase")
    repo_name: str = Field(description="Name of the indexed repository")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class QueryResponse(BaseModel):
    answer: str
    sources: list["SourceChunk"]
    confidence: float = Field(ge=0.0, le=1.0)


class SourceChunk(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    content: str
    relevance_score: float


QueryResponse.model_rebuild()
