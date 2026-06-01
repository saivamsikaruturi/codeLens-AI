"""LLM generation layer using Google Gemini API with streaming support."""

import json
import urllib.request
import urllib.error
from collections.abc import Generator

SYSTEM_PROMPT = """You are CodeLens AI, an expert code analysis assistant. You answer questions about codebases using retrieved source code chunks as context.

Rules:
1. ONLY answer based on the provided code context. If the context doesn't contain enough information, say so.
2. Always cite specific files and line numbers when referencing code.
3. Explain code clearly — describe what it does, why it's structured that way, and how it connects to other parts.
4. When discussing architecture or data flow, trace the path through the actual code.
5. If asked about something not in the context, suggest what files or areas the user might want to look at."""

QUERY_TEMPLATE = """## Retrieved Code Context

{context}

---

## User Question

{question}

---

Provide a clear, detailed answer based on the code context above. Reference specific files and line numbers."""


class CodeLensGenerator:
    """Generates answers using Google Gemini API with retrieved code context."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[dict],
        max_tokens: int = 2048,
    ) -> dict:
        context = self._format_context(retrieved_chunks)
        user_message = QUERY_TEMPLATE.format(context=context, question=question)

        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{SYSTEM_PROMPT}\n\n{user_message}"}
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.3,
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"Gemini API error ({e.code}): {error_body}")

        answer_text = result["candidates"][0]["content"]["parts"][0]["text"]

        sources = [
            {
                "file_path": chunk["metadata"]["file_path"],
                "start_line": chunk["metadata"]["start_line"],
                "end_line": chunk["metadata"]["end_line"],
                "content": chunk["content"][:200],
                "relevance_score": chunk["relevance_score"],
            }
            for chunk in retrieved_chunks
        ]

        avg_relevance = (
            sum(c["relevance_score"] for c in retrieved_chunks) / len(retrieved_chunks)
            if retrieved_chunks
            else 0
        )

        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": round(min(avg_relevance * 1.2, 1.0), 3),
        }

    def generate_stream(
        self,
        question: str,
        retrieved_chunks: list[dict],
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """Stream response tokens from Gemini API via SSE."""
        context = self._format_context(retrieved_chunks)
        user_message = QUERY_TEMPLATE.format(context=context, question=question)

        url = f"{self.base_url}/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{SYSTEM_PROMPT}\n\n{user_message}"}
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.3,
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"Gemini API error ({e.code}): {error_body}")

        buffer = ""
        for line in resp:
            decoded = line.decode("utf-8").strip()
            if decoded.startswith("data: "):
                json_str = decoded[6:]
                try:
                    chunk_data = json.loads(json_str)
                    parts = chunk_data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    for part in parts:
                        if "text" in part:
                            yield part["text"]
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue

        resp.close()

    def _format_context(self, chunks: list[dict]) -> str:
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            header = (
                f"### Chunk {i}: {meta['file_path']} "
                f"(lines {meta['start_line']}-{meta['end_line']}, "
                f"type: {meta['chunk_type']})"
            )
            formatted.append(f"{header}\n```\n{chunk['content']}\n```")
        return "\n\n".join(formatted)
