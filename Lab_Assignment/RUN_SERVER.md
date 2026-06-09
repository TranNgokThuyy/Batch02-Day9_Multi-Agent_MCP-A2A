# Hướng dẫn chạy bản server — Day08 RAG Pipeline

Bản này được làm theo dữ liệu đã cung cấp:

- 3 văn bản pháp luật PDF đã convert sang Markdown trong `data/standardized/legal/`
- 6 bài báo bạn gửi link đã đưa vào `data/landing/news/` và `data/standardized/news/`
- Task 4–10 dùng kiến trúc server: Weaviate + BM25 + reranking + PageIndex fallback + generation citation

## 1. Cài môi trường

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Tạo `.env`

```powershell
copy .env.example .env
```

Điền key nếu có:

```env
OPENAI_API_KEY=your_openai_key
JINA_API_KEY=your_jina_key
PAGEINDEX_API_KEY=your_pageindex_key
PAGEINDEX_DOC_ID=your_pageindex_doc_id
```

`JINA_API_KEY`, `PAGEINDEX_API_KEY`, `PAGEINDEX_DOC_ID` có thể để trống. Code vẫn chạy bằng fallback.

## 3. Chạy Weaviate bằng Docker

```powershell
docker compose up -d
```

Kiểm tra:

```powershell
docker ps
curl http://localhost:8080/v1/.well-known/ready
```

Nếu trả về `true` là Weaviate đã sẵn sàng.

Nếu chưa cài Docker, có thể bỏ qua bước này. Code sẽ tạo `data/local_index.json` để test local, nhưng demo server tốt nhất nên bật Docker.

## 4. Index dữ liệu

```powershell
python -m src.task4_chunking_indexing
```

Lệnh này luôn tạo local index. Nếu Weaviate đang chạy thì cũng push chunks lên Weaviate.

## 5. Chạy test từng task

```powershell
python -m src.task5_semantic_search
python -m src.task6_lexical_search
python -m src.task7_reranking
python -m src.task8_pageindex_vectorless
python -m src.task9_retrieval_pipeline
python -m src.task10_generation
```

## 6. Chạy test chấm điểm

```powershell
pytest tests/ -v
```

## 7. Chạy chatbot

```powershell
streamlit run app.py
```

Hoặc:

```powershell
python -m streamlit run app.py
```

## 8. Tạo PageIndex DOC ID

Vào dashboard PageIndex:

1. `API Keys` → tạo API key → dán vào `PAGEINDEX_API_KEY`
2. `Documents` → upload PDF/Markdown → mở document → copy `doc_id`/Document ID
3. Dán vào `.env`:

```env
PAGEINDEX_DOC_ID=doc_xxxxxxxxx
```

Task 8 chỉ là fallback. Nếu chưa có PageIndex thì pipeline vẫn chạy bằng hybrid retrieval.
