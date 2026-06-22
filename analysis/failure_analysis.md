# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Cá nhân  
**Thành viên:** NguyenBuiTanDung → M1 + M2 + M3 + M4 + M5

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.0000 | 0.7105 | +0.7105 |
| Answer Relevancy | 0.0000 | NaN | N/A |
| Context Precision | 0.0000 | 0.6583 | +0.6583 |
| Context Recall | 0.0000 | 0.6417 | +0.6417 |

Ghi chú: Điểm `Answer Relevancy` bị `NaN` do API Gemini trả về 501 Unimplemented khi RAGAS cố gọi tính điểm tương đồng. Các điểm số khác là điểm thực tế chạy từ Pipeline.

## Bottom-5 Failures

### #1
- **Question:** Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?
- **Expected:** 3 ngày làm việc có lương, không trừ phép năm.
- **Got:** Chunk về `Cam kết hoàn chi` của chính sách đào tạo.
- **Worst metric:** Context Precision
- **Error Tree:** Output sai → Context đúng? Không → Query OK? Chưa đủ vì hybrid search kéo nhầm chunk giàu từ "nhân viên" và "nghỉ việc" → Rerank chỉ dùng overlap token nên không sửa được.
- **Root cause:** Fallback dense encoder dạng hashing và fallback reranker lexical không hiểu intent "kết hôn"; retrieval bị nhiễu bởi tài liệu đào tạo có từ khóa gần giống.
- **Suggested fix:** Bật dense encoder thật + cross-encoder thật, thêm boost cho section có từ khóa quan hệ gia đình/special leave, và đưa metadata `category=leave` vào retrieval filter.

### #2
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** 15 ngày theo bản v2024 hiện hành, bản v2023 chỉ là chính sách cũ.
- **Got:** Chunk `Nghỉ phép đặc biệt` nói về kết hôn, tang lễ, sinh con.
- **Worst metric:** Context Recall
- **Error Tree:** Output sai → Context đúng? Không đầy đủ → Query OK? Có, nhưng top chunks không chứa section annual leave hiện hành → Version handling yếu.
- **Root cause:** Chunking theo parent-child chưa thêm tín hiệu version recency; search nhặt đúng chủ đề "nghỉ phép" nhưng sai subtype.
- **Suggested fix:** Thêm metadata `section`, `source`, `effective_version`, rồi ưu tiên chunk có `nghi_phep_nam_v2024` khi query hỏi policy hiện hành.

### #3
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Từ 3 năm trở lên, cộng 1 ngày cho mỗi 3 năm theo v2024.
- **Got:** Chính sách cũ: từ 5 năm trở lên, cộng 1 ngày cho mỗi 5 năm.
- **Worst metric:** Answer Relevancy
- **Error Tree:** Output sai → Context đúng? Có nhưng là context cũ → Query OK? Chưa có disambiguation "hiện hành" → Version conflict không được xử lý.
- **Root cause:** Pipeline chưa có logic resolve tài liệu superseded/current; query không được rewrite để ưu tiên bản mới.
- **Suggested fix:** Trong enrichment/metadata cần đánh dấu `status=current|superseded`, và reranker nên tăng điểm cho document mới hơn khi câu hỏi không chỉ rõ version.

### #4
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** 120 ngày theo mật khẩu v2.0 hiện hành.
- **Got:** 90 ngày theo chính sách cũ.
- **Worst metric:** Context Recall
- **Error Tree:** Output sai → Context đúng? Một phần đúng nhưng outdated → Query OK? Có → Retrieval không phân biệt version.
- **Root cause:** Search lấy `mat_khau_v1` thay vì `mat_khau_v2`; fallback reranker dựa trên lexical overlap nên hai phiên bản gần như hòa nhau.
- **Suggested fix:** Dùng metadata precedence theo filename/version, ví dụ tự động ưu tiên `v2`, `v2024`, `current`, và giảm điểm tài liệu chứa từ khóa `cũ`, `superseded`, `đã thay thế`.

### #5
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 18 ngày phép và lương 20-35 triệu VNĐ/tháng.
- **Got:** Chunk về nghỉ phép không lương tối đa 30 ngày/năm.
- **Worst metric:** Context Recall
- **Error Tree:** Output sai → Context đúng? Không → Query OK? Chưa vì đây là câu multi-hop cần ghép phép năm + thâm niên + salary band → Retrieval không kéo đủ 2 nguồn liên quan.
- **Root cause:** Pipeline hiện trả 1 chunk đầu tiên khi không bật answer LLM; không có bước tổng hợp multi-hop giữa nhiều context.
- **Suggested fix:** Với câu có nhiều slot thông tin, giữ top-3 context và thêm answer composer rule-based hoặc LLM offline/online để tổng hợp từ nhiều đoạn.

## Case Study (cho presentation)

**Question chọn phân tích:** Bao lâu phải đổi mật khẩu một lần?

**Error Tree walkthrough:**
1. Output đúng? → Không, pipeline trả `90 ngày`.
2. Context đúng? → Có ngữ cảnh về password cycle nhưng là bản `v1` đã cũ.
3. Query rewrite OK? → Chưa tốt, câu hỏi không được bổ sung tín hiệu "chính sách hiện hành".
4. Fix ở bước: Retrieval + metadata precedence + reranking theo version.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm `effective_date/current_flag` vào metadata từ tên file và heading.
- Cho hybrid search ưu tiên tài liệu current trước khi fuse.
- Thêm latency breakdown để xem M2 hay M3 đang là bottleneck.
