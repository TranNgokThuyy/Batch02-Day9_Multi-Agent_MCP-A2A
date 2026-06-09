"""Task 9 — Complete retrieval pipeline.
Semantic + lexical search -> merge -> rerank -> PageIndex fallback.
"""
from __future__ import annotations

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank
from src.task8_pageindex_vectorless import pageindex_search


def make_key(item: dict) -> str:
    meta = item.get("metadata", {})
    return f"{meta.get('path', '')}::{meta.get('chunk_id', '')}::{item.get('content', '')[:40]}"


def normalize(results: list[dict]) -> list[dict]:
    if not results:
        return []
    max_score = max(float(r.get("score", 0.0)) for r in results)
    if max_score <= 0:
        return results
    return [{**r, "score": float(r.get("score", 0.0)) / max_score} for r in results]


def merge_results(semantic_results: list[dict], lexical_results: list[dict]) -> list[dict]:
    semantic_results = normalize(semantic_results)
    lexical_results = normalize(lexical_results)
    merged: dict[str, dict] = {}

    for r in semantic_results:
        key = make_key(r)
        merged[key] = {
            "content": r["content"],
            "score": 0.6 * float(r.get("score", 0.0)),
            "source": "hybrid",
            "metadata": {**r.get("metadata", {}), "retrieval_method": "hybrid_semantic"},
        }

    for r in lexical_results:
        key = make_key(r)
        if key in merged:
            merged[key]["score"] += 0.4 * float(r.get("score", 0.0))
            merged[key]["metadata"]["retrieval_method"] = "hybrid_semantic_bm25"
        else:
            merged[key] = {
                "content": r["content"],
                "score": 0.4 * float(r.get("score", 0.0)),
                "source": "hybrid",
                "metadata": {**r.get("metadata", {}), "retrieval_method": "hybrid_bm25"},
            }

    results = list(merged.values())
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def retrieve(query: str, top_k: int = 5, score_threshold: float = 0.3) -> list[dict]:
    if top_k <= 0:
        return []
    semantic_results = semantic_search(query, top_k=10)
    lexical_results = lexical_search(query, top_k=10)
    merged = merge_results(semantic_results, lexical_results)
    reranked = rerank(query, merged, top_k=top_k)

    if not reranked or float(reranked[0].get("score", 0.0)) < score_threshold:
        pageindex_results = pageindex_search(query, top_k=top_k)
        if pageindex_results:
            return pageindex_results[:top_k]

    return reranked[:top_k]


if __name__ == "__main__":
    for r in retrieve("An Tây liên quan đến ma túy như thế nào?", top_k=5):
        print(r["score"], r["source"], r["metadata"].get("source"))
