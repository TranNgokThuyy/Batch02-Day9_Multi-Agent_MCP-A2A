"""Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Có hỗ trợ MarkItDown nếu đã cài. Nếu chưa cài, dùng fallback text extraction cơ bản cho JSON/PDF demo.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _safe_markitdown_convert(filepath: Path) -> str:
    try:
        from markitdown import MarkItDown  # type: ignore
        return MarkItDown().convert(str(filepath)).text_content
    except Exception:
        raw = filepath.read_bytes().decode("utf-8", errors="ignore")
        raw = raw.replace("%PDF-1.4", "").replace("%%EOF", "")
        raw = re.sub(r"[^\S\n]+", " ", raw)
        return raw.strip()


def convert_legal_docs():
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not legal_dir.exists():
        return []
    outputs = []
    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            text = _safe_markitdown_convert(filepath)
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(f"# {filepath.stem}\n\n{text}\n", encoding="utf-8")
            outputs.append(output_path)
    return outputs


def convert_news_articles():
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not news_dir.exists():
        return []
    outputs = []
    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            data = json.loads(filepath.read_text(encoding="utf-8"))
            header = f"# {data.get('title', 'Unknown')}\n\n**Source:** {data.get('url', 'N/A')}\n**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(header + data.get("content_markdown", ""), encoding="utf-8")
            outputs.append(output_path)
    return outputs


def convert_all():
    legal = convert_legal_docs()
    news = convert_news_articles()
    print(f"✓ Converted {len(legal)} legal docs and {len(news)} news articles to {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
