"""Cross-encoder reranker using Gemini for improved retrieval precision.

Retrieves top_k * 2 candidates from vector search, then uses the LLM
to score relevance of each chunk to the query. Returns top_k best.
"""

import json
import urllib.request
import urllib.error


class GeminiReranker:
    """Uses Gemini to rerank retrieved chunks by relevance to the query."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        if len(chunks) <= top_k:
            return chunks

        prompt = self._build_rerank_prompt(query, chunks)
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 256,
                "temperature": 0.0,
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            response_text = result["candidates"][0]["content"]["parts"][0]["text"]
            ranked_indices = self._parse_ranking(response_text, len(chunks))

            reranked = []
            for idx in ranked_indices[:top_k]:
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx].copy()
                    chunk["reranked"] = True
                    reranked.append(chunk)

            if len(reranked) < top_k:
                seen = set(ranked_indices[:top_k])
                for i, chunk in enumerate(chunks):
                    if i not in seen and len(reranked) < top_k:
                        reranked.append(chunk)

            return reranked

        except (urllib.error.HTTPError, Exception):
            return chunks[:top_k]

    def _build_rerank_prompt(self, query: str, chunks: list[dict]) -> str:
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            preview = chunk["content"][:150].replace("\n", " ")
            chunk_summaries.append(f"[{i}] {meta['file_path']}:{meta['start_line']} — {preview}")

        return f"""Given this question: "{query}"

Rank these code chunks by relevance (most relevant first). Return ONLY a JSON array of indices.

Chunks:
{chr(10).join(chunk_summaries)}

Return format: [most_relevant_index, second_most, ...]
Return ONLY the JSON array, nothing else."""

    def _parse_ranking(self, text: str, max_len: int) -> list[int]:
        text = text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                indices = json.loads(text[start:end])
                return [i for i in indices if isinstance(i, int) and 0 <= i < max_len]
            except json.JSONDecodeError:
                pass
        return list(range(max_len))
