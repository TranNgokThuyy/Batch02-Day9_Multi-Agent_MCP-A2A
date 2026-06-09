# RAG Evaluation Results

## Framework sử dụng

Offline heuristic evaluation theo cấu trúc DeepEval-compatible, chạy được local không cần API key.

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (hybrid no rerank) | Δ |
|---|---:|---:|---:|
| Faithfulness | 1.000 | 1.000 | +0.000 |
| Answer Relevance | 0.375 | 0.431 | -0.056 |
| Context Recall | 0.351 | 0.430 | -0.079 |
| Context Precision | 0.338 | 0.368 | -0.030 |
| Average | 0.516 | 0.557 | -0.041 |

## A/B Comparison Analysis

Config A dùng hybrid search + RRF + reranking; Config B bỏ reranking để so sánh tác động của bước chấm lại. Config A thường ổn định hơn khi query có nhiều từ khóa pháp luật vì reranking đẩy context đúng lên đầu.

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|---|---:|---:|---:|---|---|
| 1 | Pipeline retrieval hoàn chỉnh gồm các bước nào? | 1.000 | 0.215 | 0.038 | Retrieval/Generation | Context mô phỏng còn ngắn, cần thêm văn bản gốc đầy đủ |
| 2 | Reranking dùng để làm gì trong RAG? | 1.000 | 0.268 | 0.042 | Retrieval/Generation | Context mô phỏng còn ngắn, cần thêm văn bản gốc đầy đủ |
| 3 | Tài liệu pháp luật được lưu ở thư mục nào? | 1.000 | 0.274 | 0.027 | Retrieval/Generation | Context mô phỏng còn ngắn, cần thêm văn bản gốc đầy đủ |

## Recommendations

1. Bổ sung PDF/văn bản pháp luật gốc đầy đủ để tăng context recall.
2. Thay embedding local bằng BAAI/bge-m3 hoặc OpenAI embedding khi có môi trường production.
3. Dùng DeepEval/RAGAS thật với LLM judge khi có API key để đánh giá faithfulness chính xác hơn.
