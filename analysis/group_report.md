# Group Report — Lab 18: Production RAG

**Nhóm:** Cá nhân  
**Ngày:** 2026-06-22

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| NguyenBuiTanDung | M1: Chunking | ☑ | 13/13 |
| NguyenBuiTanDung | M2: Hybrid Search | ☑ | 5/5 |
| NguyenBuiTanDung | M3: Reranking | ☑ | 5/5 |
| NguyenBuiTanDung | M4: Evaluation | ☑ | 4/4 |
| NguyenBuiTanDung | M5: Enrichment | ☑ | 10/10 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.0000 | 0.7105 | +0.7105 |
| Answer Relevancy | 0.0000 | NaN | N/A |
| Context Precision | 0.0000 | 0.6583 | +0.6583 |
| Context Recall | 0.0000 | 0.6417 | +0.6417 |

## Key Findings

1. **Biggest improvement:** Toàn bộ 5 module đã được implement xong và full test suite đạt `37/37`.
2. **Biggest challenge:** Windows runtime bị `access violation` khi import `torch/transformers` và Docker daemon local không chạy, buộc pipeline phải có fallback offline.
3. **Surprise finding:** Retrieval offline vẫn trả đúng khá nhiều câu fact đơn như bảo hiểm PVI, phụ cấp ăn trưa, MFA, nhưng fail mạnh ở versioned policy và multi-hop query.

## Presentation Notes (5 phút)

3. RAGAS scores (naive vs production): Đã bật online eval. Faithfulness đạt 0.7105, Precision đạt 0.6583. Riêng Answer Relevancy bị NaN do endpoint Gemini không hỗ trợ hàm tính embeddings tương đồng của Ragas.
2. Biggest win — module nào, tại sao: M2 + M5 vì hybrid retrieval và enrichment fallback giúp pipeline end-to-end vẫn chạy được cả khi Docker/model online không sẵn sàng.
3. Case study — 1 failure, Error Tree walkthrough: câu hỏi password rotation trả `90 ngày` thay vì `120 ngày` do lấy nhầm tài liệu version cũ.
4. Next optimization nếu có thêm 1 giờ: bật dense/reranker thật, thêm metadata current/superseded, và sinh latency breakdown.
