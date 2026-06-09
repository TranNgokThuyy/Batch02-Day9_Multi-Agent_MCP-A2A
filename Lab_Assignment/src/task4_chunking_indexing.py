"""Task 4 — Chunking & Indexing.

Server mode: index Markdown chunks into Weaviate when Weaviate is running.
Safety mode: always write a local JSON index too, so tests and demo can still run
if Docker/Weaviate is not reachable on the current machine.

Chunking strategy:
- RecursiveCharacterTextSplitter
- chunk_size = 800, overlap = 120
Reason: Vietnamese legal text has long articles/clauses; 800 chars keeps enough
context, while 120 overlap reduces boundary information loss.

Embedding model:
- Prefer sentence-transformers/all-MiniLM-L6-v2, dimension = 384.
- If the model cannot be loaded, fallback to deterministic hash embedding,
  dimension = 384, so the project remains runnable.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    RecursiveCharacterTextSplitter = None

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
LOCAL_INDEX_PATH = PROJECT_ROOT / "data" / "local_index.json"

COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION", "RAGDocument")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

_model = None


def hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministic local embedding fallback."""
    vector = [0.0] * dim
    tokens = text.lower().replace("\n", " ").split()
    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vector[h % dim] += 1.0
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector] if norm else vector


def embed_text(text: str) -> list[float]:
    """Return embedding vector. Prefer sentence-transformers, fallback to hash."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception:
            _model = False
    if _model:
        return _model.encode(text).tolist()
    return hash_embedding(text)


def get_weaviate_client():
    """Connect to Weaviate Cloud if API key is set, otherwise local Docker."""
    try:
        import weaviate
        from weaviate.classes.init import Auth
    except Exception as exc:
        raise RuntimeError("Chưa cài weaviate-client. Chạy: pip install weaviate-client") from exc

    url = os.getenv("WEAVIATE_URL", "http://localhost:8080").strip()
    api_key = os.getenv("WEAVIATE_API_KEY", "").strip()

    if api_key:
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key),
        )

    host = os.getenv("WEAVIATE_HOST", "localhost").strip()
    port = int(os.getenv("WEAVIATE_PORT", "8080"))
    grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

    # Important: current weaviate-client versions do NOT accept grpc_host here.
    return weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)


def load_documents() -> list[dict[str, Any]]:
    """Load all standardized Markdown documents."""
    docs: list[dict[str, Any]] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        rel = md_file.relative_to(PROJECT_ROOT)
        docs.append({
            "content": content,
            "source": md_file.stem,
            "path": str(rel).replace("\\", "/"),
            "doc_type": md_file.parent.name,
            "year": infer_year(md_file.name, content),
        })
    return docs


def infer_year(filename: str, content: str) -> str:
    for y in ["2026", "2025", "2024", "2023", "2022", "2021", "2015"]:
        if y in filename or y in content[:1000]:
            return y
    return "2026"


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split documents into chunks with metadata."""
    chunks: list[dict[str, Any]] = []
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
        )
        split = splitter.split_text
    else:
        def split(text: str) -> list[str]:
            step = CHUNK_SIZE - CHUNK_OVERLAP
            return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), step)]

    for doc in documents:
        for i, text in enumerate(split(doc["content"])):
            text = text.strip()
            if not text:
                continue
            chunks.append({
                "id": f"{doc['source']}_{i}",
                "content": text,
                "source": doc["source"],
                "score": 0.0,
                "metadata": {
                    "source": doc["source"],
                    "path": doc["path"],
                    "chunk_id": i,
                    "doc_type": doc["doc_type"],
                    "year": doc["year"],
                    "chunking": "RecursiveCharacterTextSplitter",
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "embedding_model": EMBEDDING_MODEL_NAME,
                    "embedding_dim": EMBEDDING_DIM,
                },
            })
    return chunks


def write_local_index(chunks: list[dict[str, Any]]) -> None:
    LOCAL_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in chunks:
        rows.append({
            "id": c["id"],
            "content": c["content"],
            "embedding": embed_text(c["content"]),
            "source": c["source"],
            "metadata": c["metadata"],
        })
    LOCAL_INDEX_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def create_weaviate_schema(client) -> None:
    from weaviate.classes.config import Configure, DataType, Property
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="path", data_type=DataType.TEXT),
            Property(name="chunk_id", data_type=DataType.INT),
            Property(name="doc_type", data_type=DataType.TEXT),
            Property(name="year", data_type=DataType.TEXT),
        ],
    )


def index_to_weaviate(chunks: list[dict[str, Any]]) -> int:
    client = get_weaviate_client()
    try:
        create_weaviate_schema(client)
        collection = client.collections.get(COLLECTION_NAME)
        with collection.batch.dynamic() as batch:
            for c in chunks:
                meta = c["metadata"]
                batch.add_object(
                    uuid=str(uuid.uuid4()),
                    properties={
                        "content": c["content"],
                        "source": meta["source"],
                        "path": meta["path"],
                        "chunk_id": int(meta["chunk_id"]),
                        "doc_type": meta["doc_type"],
                        "year": meta["year"],
                    },
                    vector=embed_text(c["content"]),
                )
        return len(chunks)
    finally:
        client.close()


def build_index(push_to_weaviate: bool = True) -> int:
    docs = load_documents()
    chunks = chunk_documents(docs)
    write_local_index(chunks)

    if push_to_weaviate:
        try:
            total = index_to_weaviate(chunks)
            print(f"Indexed {total} chunks to Weaviate collection {COLLECTION_NAME}")
        except Exception as exc:
            print(f"[WARN] Không kết nối được Weaviate, đã lưu local index tại {LOCAL_INDEX_PATH}: {exc}")
    return len(chunks)


def run_pipeline() -> None:
    total = build_index(push_to_weaviate=True)
    print(f"Total chunks prepared: {total}")


if __name__ == "__main__":
    run_pipeline()
