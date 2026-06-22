# Reflection — NguyenBuiTanDung

**Tên:** NguyenBuiTanDung  
**Ngày:** 2026-06-22

---

## 1. Lecture Mapping

| Lecture Concept | Module | Hàm cụ thể | Observation |
|----------------|--------|-----------|-------------|
| Semantic chunking | M1 | `chunk_semantic()` | Trong local run mình để fallback heuristic mặc định để tránh crash `torch` trên Windows, nhưng interface vẫn giữ đúng semantic chunking contract. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Parent-child hoạt động ổn định và là flow chính của `src/pipeline.py`. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | Mình thêm fallback hashing encoder và Qdrant `:memory:` để pipeline vẫn chạy khi Docker daemon chưa lên. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Default local run dùng lexical rerank fallback; khi bật `USE_HEAVY_MODELS=1` có thể chuyển sang model thật. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Mình chặn online eval mặc định bằng `ENABLE_RAGAS` để tránh API/network failure làm gãy bài. |
| Contextual enrichment | M5 | `_enrich_single_call()`, `contextual_prepend()`, `extract_metadata()` | Combined mode đã implement, nhưng local default dùng heuristic fallback để giữ end-to-end run nhanh và ổn định. |

## 2. Khó khăn và cách giải quyết

- **Lỗi 1:** `UnicodeEncodeError: 'charmap' codec can't encode characters...`
  Cách xử lý: chạy các command bằng `PYTHONIOENCODING='utf-8'` và `python -X utf8` để tương thích đường dẫn/thông báo Unicode trên Windows.

- **Lỗi 2:** `Windows fatal exception: access violation` khi import `torch` qua `sentence-transformers` hoặc import `underthesea`.
  Cách xử lý: chuyển mặc định sang heuristic fallback, chỉ dùng model nặng khi chủ động bật `USE_HEAVY_MODELS=1`.

- **Lỗi 3:** Docker không lên được với thông báo `failed to connect to dockerDesktopLinuxEngine`.
  Cách xử lý: thêm fallback `QdrantClient(location=":memory:")` để DenseSearch vẫn index/search được mà không phụ thuộc container local.

- **Lỗi 4:** `APIConnectionError(Connection error.)` ở RAGAS và OpenAI answer generation.
  Cách xử lý: dùng cờ `ENABLE_RAGAS`, `ENABLE_OPENAI_ENRICHMENT`, `ENABLE_OPENAI_ANSWER`; mặc định local run không gọi online API.

## 3. Action Plan cho project cá nhân

**Project:** Internal policy assistant / enterprise RAG chatbot

### Hiện tại
- RAG pipeline hiện tại: retrieval cơ bản, chưa có version-aware metadata và chưa có fallback strategy rõ ràng.
- Known issues: hay lấy nhầm policy cũ, câu multi-hop yếu, local dev phụ thuộc quá nhiều vào external services.

### Plan áp dụng
1. [ ] Chunking strategy: dùng hierarchical chunking làm default, thêm structure-aware cho markdown/policy files.
2. [ ] Search: giữ BM25 + dense hybrid; bổ sung metadata filter theo `category`, `version`, `status=current`.
3. [ ] Reranking: bật cross-encoder ở môi trường ổn định; local dev giữ lexical fallback để test nhanh.
4. [ ] Evaluation: dùng RAGAS khi có API/network ổn định, còn local thì chạy manual failure set + regression queries.
5. [ ] Enrichment: giữ combined enrichment cho production, nhưng phải có heuristic fallback để không block deploy/dev.

### Timeline
- Tuần 1: thêm metadata version/current/superseded và sửa retrieval precedence.
- Tuần 2: bật reranker thật, benchmark latency từng bước.
- Tuần 3: chạy RAGAS online trên curated eval set và tối ưu câu multi-hop.

## 4. Nếu làm lại

- Mình sẽ tách rõ `offline-safe path` và `online-optimized path` ngay từ đầu thay vì để runtime phát hiện muộn.
- Mình muốn thử tiếp version-aware reranking và answer synthesis từ nhiều context thay vì fallback `answer = contexts[0]`.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Problem solving | 5 |
| Debugging trên Windows | 5 |
