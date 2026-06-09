"""Task 7 — Reranking.
Use Jina reranker API if configured, otherwise local keyword-overlap reranker.
"""
from __future__ import annotations

import os
import requests
from dotenv import load_dotenv

load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY", "").strip()


def keyword_overlap_score(query: str, text: str) -> float:
    q = set(query.lower().split())
    if not q:
        return 0.0
    t = set(text.lower().split())
    return len(q & t) / len(q)


def local_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    reranked = []
    for c in candidates:
        old_score = float(c.get("score", 0.0))
        overlap = keyword_overlap_score(query, c.get("content", ""))
        final = 0.7 * old_score + 0.3 * overlap
        reranked.append({
            "content": c["content"],
            "score": float(final),
            "source": c.get("source", "hybrid"),
            "metadata": {**c.get("metadata", {}), "reranker": "local_keyword_overlap"},
        })
    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if top_k <= 0 or not candidates:
        return []
    if not JINA_API_KEY:
        return local_rerank(query, candidates, top_k)
    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("results", []):
            idx = item["index"]
            c = candidates[idx]
            results.append({
                "content": c["content"],
                "score": float(item.get("relevance_score", 0.0)),
                "source": c.get("source", "hybrid"),
                "metadata": {**c.get("metadata", {}), "reranker": "jina-reranker-v2-base-multilingual"},
            })
        return results[:top_k]
    except Exception:
        return local_rerank(query, candidates, top_k)


if __name__ == "__main__":
    docs = [
        {"content": "Tàng trữ trái phép chất ma túy bị xử lý hình sự", "score": 0.7, "metadata": {}},
        {"content": "Tin tức nghệ sĩ", "score": 0.5, "metadata": {}},
    ]
    print(rerank("hình phạt ma túy", docs, top_k=2))
