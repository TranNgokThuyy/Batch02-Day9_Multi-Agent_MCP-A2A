"""Task 10 — Generation with citations.
Use OpenAI if OPENAI_API_KEY exists; otherwise extractive answer from retrieved context.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

from src.task9_retrieval_pipeline import retrieve

load_dotenv()

SYSTEM_PROMPT = """Bạn là trợ lý RAG trả lời câu hỏi dựa trên context được cung cấp.
Quy tắc:
1. Chỉ dùng thông tin có trong context.
2. Mỗi câu khẳng định sự thật phải có citation ngay sau câu.
3. Citation dùng dạng [Nguồn, Năm].
4. Nếu context không đủ bằng chứng, trả lời: I cannot verify this information.
5. Không bịa thêm thông tin ngoài context.
"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Place important chunks at beginning and end to reduce lost-in-the-middle."""
    if len(chunks) <= 2:
        return chunks
    front, back = [], []
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            front.append(chunk)
        else:
            back.insert(0, chunk)
    return front + back


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        source = meta.get("source", "Unknown")
        year = meta.get("year", "2026")
        parts.append(f"[Context {i}]\nSource: {source}\nYear: {year}\nContent:\n{c.get('content', '')}")
    return "\n\n".join(parts)


def citation(meta: dict) -> str:
    return f"[{meta.get('source', 'Unknown')}, {meta.get('year', '2026')}]"


def extractive_answer(query: str, chunks: list[dict]) -> str:
    q = set(query.lower().split())
    sentences = []
    for c in reorder_for_llm(chunks):
        meta = c.get("metadata", {})
        for s in c.get("content", "").replace("\n", " ").split("."):
            s = s.strip()
            if len(s) < 20:
                continue
            if q & set(s.lower().split()):
                sentences.append(f"{s}. {citation(meta)}")
            if len(sentences) >= 4:
                break
        if len(sentences) >= 4:
            break
    if not sentences:
        return "I cannot verify this information"
    return " ".join(sentences)


def generate_with_citation(query: str, context_chunks: list[dict] | None = None) -> dict:
    if context_chunks is None:
        context_chunks = retrieve(query, top_k=5, score_threshold=0.05)
    if not context_chunks:
        return {"answer": "I cannot verify this information", "sources": []}

    ordered = reorder_for_llm(context_chunks)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            user_prompt = f"""Question:\n{query}\n\nContext:\n{format_context(ordered)}\n\nTrả lời bằng tiếng Việt, câu nào có thông tin thực tế phải kèm citation dạng [Nguồn, Năm]."""
            # temperature thấp để giảm bịa; top_p=0.8 giúp câu trả lời tự nhiên nhưng vẫn bám context.
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                top_p=0.8,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            answer = response.choices[0].message.content or "I cannot verify this information"
            return {"answer": answer, "sources": ordered}
        except Exception:
            pass

    return {"answer": extractive_answer(query, ordered), "sources": ordered}


if __name__ == "__main__":
    print(generate_with_citation("Hình phạt tàng trữ ma túy?")["answer"])
