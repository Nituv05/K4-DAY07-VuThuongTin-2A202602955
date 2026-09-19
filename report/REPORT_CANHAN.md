# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Vũ Thường Tín
**Nhóm:** T009
**Ngày:** 19/9/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> *Viết 1-2 câu:* Tức là 2 vector có độ tương đồng về hướng trong không gian vector, Ví dụ nếu xét theo biểu diễn ngữ nghĩa của 2 câu hay 2 hình ảnh thì ta có thể hiểu chúng đang có nhiều nét giống nhau. 

**Ví dụ có độ tương tự CAO:**
- Câu A: Huy thích chó Husky 
- Câu B: Huy cũng thích động vật 
- Tại sao tương đồng: Về ngữ nghĩa cũng tương đối gần nhau. 

**Ví dụ có độ tương tự THẤP:**
- Câu A: Hôm nay trời nắng quá 
- Câu B: Mưa to thế nhỉ 
- Tại sao khác: Về ngữ nghĩa khác nhau 

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> *Viết 1-2 câu:* Tại vì Cosine có scale về 1 khoảng là -1 tới 1, còn Euclidean đơn giản chỉ tính khoảng cách. 

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* (10000 - 50)//(500 - 50) 
> *Đáp án:* 23 

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> *Viết 1-2 câu: Số chunks lên là 25, muốn tăng overlap lên để tránh mất ngữ cảnh* 

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> *Viết 2-3 câu: dùng biểu thức chính quy (regex) gì để phát hiện câu? Xử lý trường hợp ngoại lệ (edge case) nào?*
Sử dụng thư viện re, và bắt các case dấu chấm hết câu như dấu '.', '!', '?', '.\n'. 

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> *Viết 2-3 câu: thuật toán hoạt động thế nào? Base case (trường hợp cơ sở) là gì?*
Thuật toán cắt đệ quy hoạt động bằng cách chia văn bản theo các ký tự phân tách từ ưu tiên cao đến thấp (đoạn \[\rightarrow \] câu \[\rightarrow \] từ). Nếu đoạn nào vượt quá chunk_size, nó sẽ tự động gọi đệ quy để cắt tiếp bằng ký tự phân tách kế tiếp. Trường hợp cơ sở (dừng đệ quy) là khi đoạn văn bản \[\le \] chunk_size hoặc hết ký tự phân tách để thử. 

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> *Viết 2-3 câu: lưu trữ thế nào? Tính độ tương tự ra sao?*
Mỗi `Document` được lưu thành một record gồm `id`, `content`, `metadata` và vector embedding. Khi tìm kiếm, query cũng được chuyển thành vector, sau đó tính dot product với các vector đã lưu và sắp xếp theo score giảm dần để lấy `top_k`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> *Viết 2-3 câu: lọc (filter) trước hay sau? Xóa bằng cách nào?*
`search_with_filter` lọc các record theo metadata trước rồi mới tính score trên tập còn lại. `delete_document` xóa tất cả chunk có `metadata['doc_id']` trùng với `doc_id` được truyền vào và trả về `True` nếu có phần tử bị xóa.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> *Viết 2-3 câu: cấu trúc prompt? Cách đưa ngữ cảnh (inject context) vào thế nào?*
Hàm `answer` lấy các chunk liên quan từ store, đánh số nguồn rồi ghép nội dung vào phần ngữ cảnh của prompt. Prompt yêu cầu mô hình chỉ trả lời từ ngữ cảnh đã truy xuất, trích dẫn nguồn và báo không đủ thông tin nếu không tìm thấy bằng chứng phù hợp.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
$ pytest tests/ -v
============================= test session starts ==============================
platform darwin -- Python 3.14.5, pytest-9.1.1, pluggy-1.6.0 -- /opt/homebrew/opt/python@3.14/bin/python3.14
cachedir: .pytest_cache
rootdir: /Users/tinvu/Documents/LAB_VINUNI/K4-L3A-Data-Foundations
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.02s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Học phí chương trình chuẩn từ 24 đến 30 triệu đồng mỗi năm. | Mỗi năm chương trình chuẩn thu khoảng 24-30 triệu đồng học phí. | cao | 0.7906 | Đúng |
| 2 | Sinh viên được vay tối đa 4 triệu đồng mỗi tháng. | Mức vay hàng tháng cao nhất dành cho sinh viên là 4 triệu đồng. | cao | 0.5549 | Đúng |
| 3 | Hộ nghèo được nhận học bổng toàn phần 100% học phí. | Học bổng cho sinh viên hộ nghèo chi trả toàn bộ học phí. | cao | 0.5703 | Đúng |
| 4 | Học phí chương trình ELITECH khoảng 35 đến 40 triệu đồng. | Hồ sơ cần có bản sao căn cước công dân. | thấp | 0.0000 | Đúng |
| 5 | Sinh viên được hỗ trợ vay vốn ngân hàng. | Chương trình kỹ sư kéo dài năm năm. | thấp | 0.0000 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> *Viết 2-3 câu:* Cặp 2 có cùng ý nghĩa nhưng điểm chỉ đạt 0.5549 vì hai câu dùng một số từ khác nhau. Điều này cho thấy TF-IDF phụ thuộc nhiều vào từ vựng trùng nhau và chưa biểu diễn quan hệ ngữ nghĩa sâu như mô hình embedding đã được huấn luyện.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

Cấu hình chạy: `RecursiveChunker(chunk_size=1600)`, 5 tài liệu, 29 chunks, TF-IDF chuẩn hóa offline và `top_k=3`.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên được vay vốn Ngân hàng Chính sách xã hội tối đa bao nhiêu mỗi tháng và trong bao nhiêu tháng mỗi năm học? | Hỗ trợ tài chính — vay vốn sinh viên | 0.2433 | Có | Tối đa 4 triệu đồng/tháng trong 10 tháng/năm học. |
| 2 | Học phí năm học 2022-2023 của chương trình chuẩn và chương trình ELITECH dự kiến là bao nhiêu? | Học phí Đại học 2022 | 0.4646 | Có | Chương trình chuẩn 24-30 triệu đồng/năm; ELITECH 35-40 triệu đồng/năm, một số chương trình khoảng 60 triệu đồng. |
| 3 | Học sinh K65 cần có hạnh kiểm 5 kỳ THPT thế nào và học bổng hỗ trợ học tập có những mức nào? | Đăng ký học bổng HTHT cho học sinh THPT | 0.2329 | Có | Hạnh kiểm 5 kỳ loại tốt; học bổng toàn phần 100% hoặc bán phần 50% học phí. |
| 4 | Theo Nghị định 179, ba mức hỗ trợ hàng tháng cho chương trình tài năng, vi mạch hoặc khoa học cơ bản, và kỹ thuật then chốt là bao nhiêu? | Học bổng Chính phủ theo Nghị định 179 | 0.3974 | Có | Ba mức là 5.500.000, 4.200.000 và 3.700.000 đồng/tháng. |
| 5 | Hộ nghèo và hộ cận nghèo được cấp học bổng hỗ trợ học tập bằng bao nhiêu phần trăm học phí? | Tiêu chí học bổng hỗ trợ học tập | 0.4900 | Có | Hộ nghèo được 100%; hộ cận nghèo hoặc hoàn cảnh đặc biệt khó khăn được 50% học phí, trước khi xét điều chỉnh. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

Với `chunk_size=800`, câu 1 và câu 4 lấy đúng tài liệu nhưng chưa lấy đủ bằng chứng nên kết quả chỉ đạt 3/5. Tăng `chunk_size` lên 1600 giúp giữ các dữ kiện liên quan trong top-3 và nâng kết quả lên 5/5. Ở câu 5, filter `audience=high-school-student` loại tài liệu hỗ trợ tài chính dành cho sinh viên khỏi top-3 và thay bằng một chunk đúng nhóm đối tượng.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *Viết 2-3 câu:* Hiện tại nhóm chưa thực hiện phần demo nên tôi chưa có kết quả của thành viên hoặc nhóm khác để so sánh. Sau demo, tôi sẽ đối chiếu chiến lược Recursive với FixedSize, Sentence và chunk theo heading trên cùng 5 câu hỏi.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
