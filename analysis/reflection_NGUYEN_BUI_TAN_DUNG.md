# Reflection: Lecture → Project — Lab 18

**Sinh viên:** Nguyễn Bùi Tấn Dũng  
**Project nhóm:** C2-App-028  
**Vai trò trong project:** Nội dung và QA — xây dựng Gold Q&A, kiểm thử UAT,
phân tích lỗi, chuẩn bị edge cases, demo set và corpus thư viện.

---

## 1. Mapping bài giảng vào project LibAssist

LibAssist là trợ lý thư viện song ngữ cho VinUniversity. Hệ thống tách câu hỏi
thành ba luồng: tìm tài liệu thật qua Primo, hỏi quy định/how-to qua RAG trên
LibGuides và từ chối khi không có bằng chứng. Sau khi đối chiếu Lab 18 với repo
project, tôi nhận thấy project đã có nền tảng RAG hoạt động nhưng vẫn còn khoảng
trống rõ ràng ở hybrid search, cross-encoder reranking và evaluation theo RAGAS.

| Lecture Concept | Module Lab | Code hiện tại trong LibAssist | Observation và bài học |
|---|---|---|---|
| Semantic, hierarchical và structure-aware chunking | M1 | `chunk_sections()` trong `apps/api/app/services/libguides_crawler.py`; `chunk_text()` trong `document_ingest.py` | LibGuides được chia theo heading rồi theo câu với giới hạn 900 ký tự; tài liệu upload được gom theo đoạn, giới hạn 800 ký tự và overlap 80. Cách này đã tận dụng cấu trúc tài liệu, nhưng chưa dùng embedding để xác định ranh giới ngữ nghĩa và chưa lưu quan hệ parent–child. Các lỗi HOW-01 và HOW-08 cho thấy một chunk đúng kích thước chưa chắc đã chứa trọn bằng chứng cần thiết. |
| BM25 + Dense fusion và RRF | M2 | `search_chunks_scored()`, `search_chunks_keyword()`, `build_retrieval_queries()` và `_merge_scored()` trong `apps/api/app/services/chat.py` | Project đang dùng dense retrieval với `text-embedding-3-small`, cosine similarity và full scan các vector JSON trong PostgreSQL. Keyword `ILIKE` chủ yếu là fallback hoặc query bổ sung, chưa phải BM25 và chưa fusion bằng RRF. Multi-query expansion cùng anchor boost đã giúp automated eval tăng mạnh, nhưng việc ghép điểm thủ công khó hiệu chỉnh và có thể bỏ sót từ khóa tiếng Việt quan trọng. |
| Cross-encoder reranking | M3 | `_anchor_boost()`, `_pick_citations()` và các kind/topic bonus trong `chat.py` | LibAssist chưa có cross-encoder. Việc rerank hiện dựa vào cosine score, metadata kind, anchor terms và ngưỡng thủ công. Human UAT phát hiện HOW-01 lấy nhầm hướng dẫn request/catalog thay vì quy trình mượn tại self-check/Circulation Desk, chứng tỏ retrieval tìm được tài liệu gần chủ đề nhưng thứ tự cuối chưa đủ chính xác. Cross-encoder phù hợp để chấm lại top 10–20 ứng viên theo cặp query–chunk. |
| RAGAS và failure analysis | M4 | `apps/api/scripts/eval_gold.py`, `Gold-QA.csv`, `Edge-W3.csv`, `WORKLOG.md` | Project có 21 Gold questions gồm 9 CAT, 8 HOW và 4 REF. Automated hints từng báo 21/21, nhưng human UAT của tôi chỉ ghi nhận 15/21 (71,4%). Đây là minh chứng rõ rằng citation count và keyword hit không đo đầy đủ faithfulness hay completeness. RAGAS có thể bổ sung Faithfulness, Answer Relevancy, Context Precision và Context Recall; human UAT vẫn cần giữ làm lớp kiểm tra cuối. |
| Contextual enrichment, metadata và hypothetical questions | M5 | `build_chunk_content()`, `build_chunk_header()`, `parse_chunk_metadata()` và `format_chunk_for_llm()` trong `corpus_categories.py` | Mỗi chunk đã được prepend các trường `Kind`, `Category`, `Resource`, `Source`, `Section` và `Use for`. Đây là một dạng contextual enrichment tốt, giúp filtering, citation và LLM hiểu vai trò của chunk. Tuy nhiên project chưa sinh summary hay hypothetical questions cho từng chunk. Metadata hiện có thể được tận dụng để tạo query expansion và index BM25 theo trường. |

### Nhận xét từ kết quả Lab 18

Trong Lab 18, production pipeline đạt Faithfulness 0,7063, Answer Relevancy
0,6936, Context Precision 0,8958 và Context Recall 0,8167. Naive baseline lại có
Faithfulness 0,8417, Answer Relevancy 0,6802, Context Precision 0,9250 và Context
Recall 0,9250. Production chỉ cải thiện nhẹ Answer Relevancy nhưng giảm ba metric
còn lại. Điều này nhắc tôi rằng thêm nhiều kỹ thuật không tự động làm RAG tốt hơn:
enrichment có thể gây nhiễu, chunk nhỏ có thể làm mất bằng chứng và reranker có
thể loại nhầm tài liệu cần cho câu hỏi nhiều ý.

Liên hệ với LibAssist, automated score 100% cũng không đồng nghĩa sản phẩm đã
hoàn hảo. Human UAT tìm ra sáu câu thiếu facts bắt buộc dù hệ thống có citation.
Vì vậy, mỗi thay đổi retrieval phải được so với baseline trên cùng test set, đồng
thời kiểm tra cả câu trả lời lẫn bằng chứng hỗ trợ.

---

## 2. Khó khăn và cách giải quyết

### 2.1 Model embedding cục bộ tải quá lâu

Khi làm M2, tiến trình tải `BAAI/bge-m3` đứng trong khoảng hai giờ và terminal
hiện đúng cảnh báo:

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

Tôi kiểm tra rằng `.venv` vẫn dùng Python 3.12 dù máy đồng thời cài Python 3.14,
vì vậy nguyên nhân không phải virtual environment mất kết nối mà là model lớn,
lần tải đầu chưa hoàn tất và request Hugging Face không có token.

**Cách debug và bài học:**

1. Kiểm tra `python --version` và executable trong `.venv` để loại trừ lỗi interpreter.
2. Tách bước tải model khỏi bước chạy pipeline để biết terminal đang tải hay treo.
3. Không chọn model chỉ dựa vào chất lượng; phải xét thời gian cold start, dung
   lượng cache và môi trường chấm bài.
4. Với LibAssist, `text-embedding-3-small` qua API phù hợp hơn cho staging hiện
   tại. Nếu chuyển sang model local, cần pre-download trong image build, cache
   model và pin phiên bản thay vì tải ở request đầu tiên.

### 2.2 Automated evaluation khác xa human UAT

`eval_gold.py` từng báo **21/21 (100%) automated hints**, nhưng khi tôi kiểm tra
từng ảnh chat và citation theo cột `Pass nếu`, kết quả thật là **15/21 (71,4%)**:
CAT 8/9, HOW 4/8 và REF 3/4.

Nguyên nhân là evaluator hiện tại chủ yếu kiểm tra:

- có citation hay không;
- citation có URL Primo/LibGuides hay không;
- có trúng một số keyword kỳ vọng hay không;
- câu trả lời có giống danh sách giả hoặc có tín hiệu refusal hay không.

Các kiểm tra này không xác nhận citation có thực sự hỗ trợ từng claim. Ví dụ,
HOW-02 và HOW-03 đã retrieve được tài liệu liên quan nhưng câu trả lời bỏ sót giới
hạn gia hạn hoặc mức phạt bắt buộc. Đây là answer-generation failure, không phải
retrieval failure.

**Cách giải quyết:** Tôi chấm lại toàn bộ 21 Gold questions bằng rubric, lưu bằng
chứng, phân loại lỗi thành `intent`, `retrieval`, `answer_generation`,
`guardrail_template` và `test_execution`. Sau đó tôi tạo thêm 10 edge cases và
chốt 5 câu demo ổn định. Việc phân loại theo layer giúp nhóm biết sửa đúng nơi
thay vì thay prompt cho mọi lỗi.

### 2.3 Corpus đúng nguồn nhưng chưa chắc retrieval-friendly

Tôi đã curate tám tài liệu thư viện, thêm provenance, ngày kiểm tra, manifest và
quality report. Corpus đã loại giờ mở cửa cũ năm 2024, ví dụ database chưa xác
minh và nội dung marketing không liên quan. Tuy nhiên HOW-01 và HOW-08 vẫn fail
vì section/chunk chưa đặt đúng bằng chứng lên đầu.

**Cách giải quyết và kiến thức bổ sung:**

- Không chỉ kiểm tra tính đúng của tài liệu; phải kiểm tra khả năng retrieve theo
  câu hỏi thật.
- Tách trang borrow/renew/return thành các section Renew, Return, Request và
  Equipment giúp retrieval tốt hơn.
- Dùng `Kind`, `Category`, `Resource`, `Source`, `Section` để tăng context cho
  embedding và citation.
- Bài Lab 18 bổ sung cho tôi cách dùng semantic/hierarchical chunking và parent
  expansion để giữ đủ ngữ cảnh khi một câu trả lời cần nhiều đoạn.

### 2.4 Phân biệt lỗi intent, retrieval và generation

Edge case `EDGE-07` kết hợp “tìm sách machine learning” với “hướng dẫn cách mượn”
nhưng hệ thống xử lý cả câu như một catalog query. Đây là lỗi decomposition ở
intent layer. Trong khi đó:

- HOW-01 là retrieval failure vì lấy sai hướng dẫn.
- HOW-02/03 là generation failure vì context đúng nhưng thiếu facts.
- REF-01 là guardrail-template failure vì từ chối an toàn nhưng chưa nói rõ lịch
  FAQ là dữ liệu năm 2024.

**Bài học:** Error Tree phải bắt đầu từ output, kiểm tra context, kiểm tra routing
rồi mới quyết định sửa chunking, search, reranking hay prompt. Nếu không, nhóm có
thể tăng top-k hoặc viết thêm prompt nhưng không chạm đúng root cause.

---

## 3. Action Plan áp dụng cho project

## Project: C2-App-028

### Hiện tại
- Dự án đang ở Tuần 1: Khởi động và thiết lập môi trường.
- Đã hoàn thành tài liệu đặc tả yêu cầu, quy trình Git và cấu trúc thư mục.
- Đang thiết lập khung dự án RAG cơ bản và dữ liệu mẫu.

### Plan áp dụng kiến thức Day 18 (RAGAS Eval, Reranking, Enrichment)

1. [ ] **Thiết lập Baseline & Evaluation (RAGAS) ngay từ đầu**
   - Xây dựng ngay một tập **Gold Q&A (10-20 câu hỏi)** đại diện cho các intents chính.
   - Tích hợp **RAGAS** vào test local để đảm bảo các thay đổi không làm giảm chất lượng hệ thống.

2. [ ] **Chiến lược Chunking & Metadata Enrichment**
   - Sử dụng **Structure-aware chunking** (giữ nguyên bảng, danh sách).
   - Thêm Contextual Prepend (Enrichment): Gắn metadata (Category, Date, Source) vào đầu mỗi chunk.

3. [ ] **Hybrid Search & Reciprocal Rank Fusion (RRF)**
   - Kết hợp Dense Retrieval và Sparse Retrieval (BM25) để bắt chính xác keyword.
   - Sử dụng RRF để kết hợp điểm số một cách tự nhiên.

4. [ ] **Tích hợp Cross-Encoder Reranker**
   - Triển khai mô hình Reranking nhỏ (ge-reranker) lọc Top-20 ứng viên xuống Top-5 context tốt nhất.

5. [ ] **Quy trình Failure Analysis liên tục**
   - Áp dụng Error Tree để phân tích lỗi: Lỗi do Chunking? Search? Rerank? Hay Generation?
   - Ghi chú lỗi vào WORKLOG.md theo từng Sprint.

### Timeline Dự kiến

| Thời gian | Mục tiêu |
|---|---|
| Sprint 1-2 | Tích hợp Naive RAG, chuẩn bị Gold Q&A, setup test RAGAS. |
| Sprint 3 | Chuyển đổi Hybrid Search (BM25 + Dense). |
| Sprint 4 | Thêm Reranker, tinh chỉnh Metadata Enrichment, Failure Analysis. |
| Sau MVP | Tự động hóa quá trình Evaluation vào pipeline CI/CD. |
