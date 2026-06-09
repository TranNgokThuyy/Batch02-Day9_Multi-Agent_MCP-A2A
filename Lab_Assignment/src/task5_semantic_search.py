"""Task 5 — Semantic Search.
Prefer Weaviate near_vector. Fallback to local JSON index if server is unavailable.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.task4_chunking_indexing import (
    COLLECTION_NAME,
    LOCAL_INDEX_PATH,
    embed_text,
    get_weaviate_client,
)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def load_local_index() -> list[dict]:
    if not LOCAL_INDEX_PATH.exists():
        from src.task4_chunking_indexing import build_index
        build_index(push_to_weaviate=False)
    return json.loads(Path(LOCAL_INDEX_PATH).read_text(encoding="utf-8"))


def semantic_search_local(query: str, top_k: int = 10) -> list[dict]:
    qvec = embed_text(query)
    results = []
    for item in load_local_index():
        score = cosine_similarity(qvec, item.get("embedding", []))
        results.append({
            "content": item["content"],
            "score": float(score),
            "source": "semantic",
            "metadata": {**item.get("metadata", {}), "retrieval_method": "semantic_local"},
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    if top_k <= 0:
        return []
    try:
        from weaviate.classes.query import MetadataQuery
        client = get_weaviate_client()
        try:
            collection = client.collections.get(COLLECTION_NAME)
            response = collection.query.near_vector(
                near_vector=embed_text(query),
                limit=top_k,
                return_metadata=MetadataQuery(distance=True),
            )
            results = []
            for obj in response.objects:
                distance = obj.metadata.distance or 0.0
                score = 1.0 / (1.0 + float(distance))
                props = obj.properties
                results.append({
                    "content": props["content"],
                    "score": float(score),
                    "source": "semantic",
                    "metadata": {
                        "source": props.get("source"),
                        "path": props.get("path"),
                        "chunk_id": props.get("chunk_id"),
                        "doc_type": props.get("doc_type"),
                        "year": props.get("year"),
                        "retrieval_method": "semantic_weaviate",
                    },
                })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        finally:
            client.close()
    except Exception:
        return semantic_search_local(query, top_k=top_k)


if __name__ == "__main__":
    for r in semantic_search("hình phạt ma túy", top_k=5):
        print(r["score"], r["metadata"].get("source"))
