"""Supervisor-Workers orchestration for the Day08 RAG agent.

The existing Day08 modules remain the source of truth for retrieval and
generation. This layer makes the multi-agent pattern explicit:
Supervisor -> Retrieval Worker -> Evidence Worker -> Generation Worker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation


FALLBACK_ANSWER = "I cannot verify this information"


@dataclass(frozen=True)
class WorkerResult:
    """Standard shape returned by each worker."""

    name: str
    status: str
    output: Any
    error: str | None = None

    def to_trace(self) -> dict:
        trace = {"worker": self.name, "status": self.status}
        if self.error:
            trace["error"] = self.error
        return trace


def retrieval_worker(query: str, top_k: int, score_threshold: float) -> WorkerResult:
    """Fetch candidate context chunks with the existing hybrid RAG pipeline."""
    try:
        sources = retrieve(query, top_k=top_k, score_threshold=score_threshold)
        return WorkerResult("retrieval_worker", "ok", sources[:top_k])
    except Exception as exc:
        return WorkerResult("retrieval_worker", "error", [], str(exc))


def evidence_worker(sources: list[dict], top_k: int) -> WorkerResult:
    """Keep only source records that can support citation-style answers."""
    validated = []
    for source in sources[:top_k]:
        if not isinstance(source, dict):
            continue
        content = str(source.get("content", "")).strip()
        if not content:
            continue
        validated.append(
            {
                "content": content,
                "score": float(source.get("score", 0.0) or 0.0),
                "source": source.get("source", "unknown"),
                "metadata": source.get("metadata", {}) or {},
            }
        )
    status = "ok" if validated else "empty"
    return WorkerResult("evidence_worker", status, validated)


def generation_worker(query: str, evidence: list[dict]) -> WorkerResult:
    """Generate the final cited response from validated evidence."""
    if not query.strip() or not evidence:
        return WorkerResult("generation_worker", "empty", {"answer": FALLBACK_ANSWER, "sources": evidence})

    try:
        result = generate_with_citation(query, evidence)
        if not isinstance(result, dict):
            result = {"answer": str(result), "sources": evidence}
        result.setdefault("sources", evidence)
        result.setdefault("answer", FALLBACK_ANSWER)
        return WorkerResult("generation_worker", "ok", result)
    except Exception as exc:
        return WorkerResult(
            "generation_worker",
            "error",
            {"answer": FALLBACK_ANSWER, "sources": evidence},
            str(exc),
        )


def supervisor_answer(query: str, top_k: int = 5, score_threshold: float = 0.05) -> dict:
    """Coordinate the Day08 RAG workers and return an app-friendly response."""
    safe_query = (query or "").strip()
    safe_top_k = max(0, int(top_k))
    worker_results: list[WorkerResult] = []

    if not safe_query or safe_top_k == 0:
        retrieval = WorkerResult("retrieval_worker", "skipped", [])
    else:
        retrieval = retrieval_worker(safe_query, safe_top_k, score_threshold)
    worker_results.append(retrieval)

    evidence = evidence_worker(retrieval.output, safe_top_k)
    worker_results.append(evidence)

    generation = generation_worker(safe_query, evidence.output)
    worker_results.append(generation)

    generated = generation.output if isinstance(generation.output, dict) else {}
    sources = generated.get("sources") or evidence.output

    return {
        "answer": generated.get("answer", FALLBACK_ANSWER),
        "sources": sources[:safe_top_k],
        "trace": {
            "pattern": "supervisor_workers",
            "supervisor": "rag_supervisor",
            "worker_steps": [result.to_trace() for result in worker_results],
            "source_count": len(sources[:safe_top_k]),
        },
    }


__all__ = [
    "supervisor_answer",
    "retrieval_worker",
    "evidence_worker",
    "generation_worker",
]
