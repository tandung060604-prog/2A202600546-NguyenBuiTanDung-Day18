# Failure Analysis — Lab 18: Production RAG

**Hình thức:** Bài làm cá nhân  
**Sinh viên:** Nguyễn Bùi Tấn Dũng

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.8417 | 0.7063 | -0.1354 |
| Answer Relevancy | 0.6802 | 0.6936 | +0.0134 |
| Context Precision | 0.9250 | 0.8958 | -0.0292 |
| Context Recall | 0.9250 | 0.8167 | -0.1083 |

Production cải thiện Answer Relevancy thêm 0.0134, nhưng giảm ở Faithfulness,
Context Precision và Context Recall so với naive baseline. Nguyên nhân chính là
enrichment làm context dài hơn, reranker chỉ giữ 3 kết quả và một số bảng hoặc
thông tin cần tổng hợp từ nhiều tài liệu bị cắt hoặc loại khỏi context cuối.
Tuy vậy, theo rubric, production vẫn có 3/4 metric đạt ngưỡng 0.70 và đủ điều kiện
nhận tối đa phần điểm RAGAS.

## Bottom-5 Failures

### #1 — Câu hỏi tổng hợp phép năm và lương

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 18 ngày phép; lương Senior từ 20–35 triệu VNĐ/tháng.
- **Got:** “Không tìm thấy.”
- **Worst metric:** Faithfulness = 0.0000.
- **Error Tree:** Output sai → context chỉ có bằng chứng về 18 ngày phép → thiếu
  chunk khung lương Senior → truy vấn yêu cầu tổng hợp hai tài liệu → lỗi retrieval
  coverage và generation từ chối toàn bộ câu trả lời.
- **Root cause:** Reranker chỉ giữ 3 context nhưng không bảo đảm phủ đủ hai ý
  “thâm niên” và “lương”; mô hình không trả lời phần đã có bằng chứng.
- **Suggested fix:** Tách truy vấn đa ý thành hai sub-query, hợp nhất kết quả,
  tăng context cuối lên 4–5 hoặc mở rộng parent context; yêu cầu mô hình trả lời
  từng ý và nêu rõ ý nào thiếu bằng chứng.

### #2 — Xung đột phiên bản chính sách mật khẩu

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** 120 ngày theo chính sách v2.0; quy định 90 ngày của v1.0 đã bị thay thế.
- **Got:** “Không tìm thấy.”
- **Worst metric:** Faithfulness = 0.0000.
- **Error Tree:** Output sai → context có cả 90 ngày và 120 ngày → query đúng →
  thiếu cơ chế ưu tiên phiên bản hiện hành → generation không giải quyết được
  xung đột và từ chối trả lời.
- **Root cause:** Metadata về phiên bản/trạng thái chưa được dùng để lọc hoặc
  rerank; prompt chưa hướng dẫn ưu tiên tài liệu mới nhất và tài liệu còn hiệu lực.
- **Suggested fix:** Trích xuất `version`, `effective_date`, `status`; loại hoặc hạ
  điểm tài liệu “ĐÃ THAY THẾ”, đồng thời thêm quy tắc ưu tiên chính sách hiện hành
  vào prompt sinh câu trả lời.

### #3 — Tổng hợp và tính lương thử việc

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** 85% × 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** “Không tìm thấy.”
- **Worst metric:** Faithfulness = 0.0000.
- **Error Tree:** Output sai → context có tỷ lệ 85% và trần lương Junior 20 triệu
  ở hai chunk khác nhau → retrieval đủ bằng chứng → generation không liên kết hai
  chunk và không thực hiện phép tính.
- **Root cause:** Prompt quá thiên về từ chối, chưa yêu cầu tổng hợp bằng chứng và
  trình bày phép tính đơn giản.
- **Suggested fix:** Dùng prompt có cấu trúc “trích số liệu → công thức → kết quả”,
  đặt temperature thấp và yêu cầu kiểm tra tất cả context trước khi kết luận không
  tìm thấy.

### #4 — Quyền lợi PVI trong thời gian thử việc

- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** Không; chỉ tham gia bảo hiểm xã hội bắt buộc.
- **Got:** “Không tìm thấy.”
- **Worst metric:** Faithfulness = 0.0000.
- **Error Tree:** Output sai → context đầu tiên chứa câu trả lời trực tiếp → query
  đúng và retrieval đúng → lỗi hoàn toàn ở bước generation.
- **Root cause:** Mô hình bỏ qua bằng chứng phủ định rõ ràng trong context, có thể
  do context enrichment dài hoặc prompt từ chối quá mạnh.
- **Suggested fix:** Đưa chunk có điểm rerank cao nhất lên đầu, giới hạn phần
  enrichment, yêu cầu trả lời Có/Không trước rồi mới giải thích bằng bằng chứng.

### #5 — Bảng phê duyệt mua sắm bị cắt

- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Tổng Giám đốc (CEO).
- **Got:** Giám đốc phòng ban.
- **Worst metric:** Context Recall = 0.0000.
- **Error Tree:** Output sai → context liên quan nhưng dòng “Trên 50 triệu” bị cắt
  ngay trước ô người phê duyệt → query đúng → chunking làm vỡ hàng của bảng →
  generation suy đoán từ ngưỡng liền trước.
- **Root cause:** Hierarchical chunking theo kích thước ký tự/token không bảo toàn
  toàn bộ bảng Markdown.
- **Suggested fix:** Áp dụng structure-aware chunking cho bảng, không tách giữa
  các ô trong cùng hàng; khi một child chứa một phần bảng, đính kèm toàn bộ bảng
  hoặc parent section trước khi rerank.

## Case Study (cho presentation)

**Question chọn phân tích:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?

**Error Tree walkthrough:**

1. Output đúng? → Không, hệ thống trả lời Director thay vì CEO.
2. Context đúng? → Đúng chủ đề nhưng thiếu ô cuối của hàng “Trên 50 triệu”.
3. Query rewrite OK? → Query rõ ràng, không cần rewrite.
4. Fix ở bước: M1 structure-aware chunking và parent-context expansion trước M3.

Đây là lỗi điển hình cho thấy context precision cao chưa đủ: hệ thống có thể tìm
đúng tài liệu nhưng vẫn thiếu đúng mẩu bằng chứng quyết định do ranh giới chunk.

**Nếu có thêm 1 giờ, sẽ optimize:**

- Bảo toàn bảng Markdown và mở rộng parent context cho câu hỏi chứa ngưỡng số.
- Thêm metadata version/status để ưu tiên chính sách hiện hành.
- Tách câu hỏi đa ý thành sub-query rồi hợp nhất bằng chứng trước generation.
- Tinh chỉnh enrichment và reranking, sau đó chạy lại cả baseline lẫn production
  trên cùng test set để kiểm chứng mức cải thiện.
