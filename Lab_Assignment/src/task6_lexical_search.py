"""Task 6 — Lexical Search.
Prefer Weaviate BM25. Fallback to rank-bm25/simple keyword scoring locally.
"""
from __future__ import annotations

from src.task4_chunking_indexing import COLLECTION_NAME, get_weaviate_client
from src.task5_semantic_search import load_local_index


def tokenize(text: str) -> list[str]:
    return text.lower().replace(",", " ").replace(".", " ").split()


def lexical_search_local(query: str, top_k: int = 10) -> list[dict]:
    index = load_local_index()
    corpus = [item["content"] for item in index]
    tokenized_corpus = [tokenize(c) for c in corpus]
    tokenized_query = tokenize(query)
    results = []
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)
    except Exception:
        q = set(tokenized_query)
        scores = []
        for tokens in tokenized_corpus:
            scores.append(float(sum(1 for t in tokens if t in q)))
    for item, score in zip(index, scores):
        results.append({
            "content": item["content"],
            "score": float(score),
            "source": "lexical",
            "metadata": {**item.get("metadata", {}), "retrieval_method": "bm25_local"},
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    if top_k <= 0:
        return []
    try:
        from weaviate.classes.query import MetadataQuery
        client = get_weaviate_client()
        try:
            collection = client.collections.get(COLLECTION_NAME)
            response = collection.query.bm25(
                query=query,
                limit=top_k,
                return_metadata=MetadataQuery(score=True),
            )
            results = []
            for obj in response.objects:
                props = obj.properties
                results.append({
                    "content": props["content"],
                    "score": float(obj.metadata.score or 0.0),
                    "source": "lexical",
                    "metadata": {
                        "source": props.get("source"),
                        "path": props.get("path"),
                        "chunk_id": props.get("chunk_id"),
                        "doc_type": props.get("doc_type"),
                        "year": props.get("year"),
                        "retrieval_method": "bm25_weaviate",
                    },
                })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        finally:
            client.close()
    except Exception:
        return lexical_search_local(query, top_k=top_k)


if __name__ == "__main__":
    for r in lexical_search("Điều 248 ma túy", top_k=5):
        print(r["score"], r["metadata"].get("source"))
