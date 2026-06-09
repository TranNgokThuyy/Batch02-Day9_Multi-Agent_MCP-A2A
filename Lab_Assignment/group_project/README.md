# Bài Tập Nhóm — Search Engine / RAG Chatbot

## Mục tiêu

Xây dựng chatbot RAG trả lời câu hỏi về **pháp luật Việt Nam về ma túy** và **tin tức liên quan đến nghệ sĩ/chất cấm**, có citation và hiển thị nguồn.

## Kiến trúc hệ thống

```text
User / Streamlit UI
        |
        v
Task 10: generate_with_citation
        |
        v
Task 9: Retrieval pipeline
        |-- Task 5: semantic search local hash embedding
        |-- Task 6: lexical BM25
        |-- Task 7: RRF merge + reranking heuristic
        |-- Task 8: PageIndex/vectorless fallback
        v
Context chunks + source metadata
        |
        v
Answer with citation
```

## Dữ liệu

- `data/landing/legal/`: file gốc văn bản pháp luật dạng PDF demo.
- `data/landing/news/`: bài báo dạng JSON có metadata `url`, `title`, `date_crawled`, `content_markdown`.
- `data/standardized/`: toàn bộ dữ liệu đã chuyển sang Markdown.

## Phân công công việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|---|---|---|---|
| Trần Ngọc Thụy | 2A202600799 | Hoàn thiện pipeline cá nhân Task 1–10 | Hoàn thành |
| Thành viên nhóm |  | Tích hợp UI Streamlit | Demo sẵn trong `app.py` |
| Thành viên nhóm |  | Evaluation pipeline | Hoàn thành bản offline heuristic |
| Thành viên nhóm |  | README + báo cáo | Hoàn thành |

## Hướng dẫn chạy

```bash
pip install -r requirements.txt
pytest tests/ -v
streamlit run app.py
```

Nếu môi trường không có API key, pipeline vẫn chạy bằng local retrieval/generation fallback.

## Evaluation

Các file deliverable nằm trong `group_project/evaluation/`:

- `golden_dataset.json`: 15 cặp Q&A.
- `eval_pipeline.py`: script đánh giá offline 4 metric.
- `results.md`: bảng điểm A/B, worst performers và đề xuất cải tiến.

Chạy:

```bash
python group_project/evaluation/eval_pipeline.py
```
