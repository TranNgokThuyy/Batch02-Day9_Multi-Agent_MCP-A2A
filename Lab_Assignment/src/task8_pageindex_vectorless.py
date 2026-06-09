"""Task 8 — PageIndex Vectorless RAG.

PageIndex dashboard hiện thường dùng Document ID (doc_id) sau khi upload tài liệu.
Điền PAGEINDEX_API_KEY và PAGEINDEX_DOC_ID trong .env nếu dùng PageIndex thật.
Nếu chưa cấu hình, function trả [] để Task 9 fallback không crash.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
PAGEINDEX_DOC_ID = os.getenv("PAGEINDEX_DOC_ID", "").strip()


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless retrieval using PageIndex. Returns standard RAG result dicts."""
    if top_k <= 0:
        return []
    if not PAGEINDEX_API_KEY or not PAGEINDEX_DOC_ID:
        return []

    try:
        # SDK PageIndex có thể thay đổi theo version. Logic chính: query theo doc_id.
        from pageindex import PageIndexClient  # type: ignore
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

        if hasattr(client, "query"):
            raw_results = client.query(doc_id=PAGEINDEX_DOC_ID, query=query, top_k=top_k)
        elif hasattr(client, "search"):
            raw_results = client.search(doc_id=PAGEINDEX_DOC_ID, query=query, top_k=top_k)
        else:
            return []
    except Exception:
        return []

    results: list[dict] = []
    for r in list(raw_results)[:top_k]:
        content = getattr(r, "text", None) or getattr(r, "content", None) or str(r)
        score = getattr(r, "score", 0.0) or 0.0
        metadata = getattr(r, "metadata", {}) or {}
        results.append({
            "content": content,
            "score": float(score),
            "source": "pageindex",
            "metadata": {
                "source": metadata.get("source", "pageindex"),
                "year": metadata.get("year", "2026"),
                "doc_id": PAGEINDEX_DOC_ID,
                "retrieval_method": "pageindex_vectorless",
            },
        })
    return results


if __name__ == "__main__":
    print(pageindex_search("ma túy", top_k=2))
