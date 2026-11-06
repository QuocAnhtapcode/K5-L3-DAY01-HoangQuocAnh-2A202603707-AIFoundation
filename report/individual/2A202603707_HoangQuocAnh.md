# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603707
- **Họ và tên:** Hoàng Quốc Anh
- **SEED** (in ở ô đầu notebook): 8387

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | Yes |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | Yes |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | Yes |

## 2. Kết quả chính

Dữ liệu sau khi bỏ trùng có 5.169 tin, gồm 4.516 ham và 653 spam. Tập train/validation/test lần lượt có 3.619/775/775 tin; SUB gồm 40 tin cân bằng, với 20 spam và 20 ham.

| Mô hình | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
|---|---:|---:|---:|---:|
| 1. BoW + LogReg | 0.943 | 1.000 | 0.074 | 0.000 |
| 2. BoW + MLP | 0.984 | 1.000 | 0.424 | 0.000 |
| 3a. MiniLM + kNN | 0.922 | 1.000 | 19.227 | 0.000 |
| 3b. MiniLM + LogReg | 0.952 | 1.000 | 7.938 | 0.000 |
| 3c. MiniLM + MLP head | 0.941 | 0.974 | 7.766 | 0.000 |
| 4a. LLM nhỏ zero-shot | 0.974* | 0.974 | 1806.872 | 0.009 |
| 4b. LLM lớn zero-shot | 0.974* | 0.974 | 1418.132 | 0.089 |
| 4c. LLM nhỏ few-shot | 0.974* | 0.974 | 1201.635 | 0.023 |

*Ở ba hàng LLM, notebook ghi lại kết quả SUB vào cả hai cột F1; chưa đánh giá LLM trên toàn bộ 775 tin test. Vì vậy, so sánh chất lượng giữa mọi mô hình phải dùng cột F1 SUB 40. LLM nhỏ là `gpt-6-luna`, LLM lớn là `gpt-6.1-terra` theo cấu hình lab.*

BoW + LogReg là điểm duy nhất trên đường Pareto của biểu đồ SUB: F1 = 1.000 và độ trễ thấp nhất. Trong lần chạy mới nhất, cả năm mô hình BoW/Embedding đều đạt p95 < 50 ms; ba mô hình LLM đều vượt SLA. SUB nhỏ nên F1 = 1.000 chưa đủ để kết luận mô hình tổng quát tốt; chi phí 0 ở các hàng chạy cục bộ chỉ có nghĩa là không tính phí API, chưa bao gồm hạ tầng.

## 3. Dự đoán và thực tế

Ba dự đoán dưới đây đều có trong các lời gọi `predict(...)` của notebook.

| Dự đoán | Tôi dự đoán gì? | Kết quả thực tế | Vì sao lệch? |
|---|---|---|---|
| `hard_eval` | Accuracy BoW khoảng 70%, Embedding khoảng 80%; LLM trên 90% và khó nhất ở prompt injection. | Hai mô hình BoW chỉ đạt 54%; kNN/Embedding + LogReg/Embedding + MLP đạt 72%/64%/66%. Hai LLM đạt 100%, kể cả nhóm injection. | Tôi đánh giá quá cao khả năng tổng quát của mô hình trên tập train. Case khó thay đổi giọng văn, che từ khóa và chứa lừa đảo ít xuất hiện trong dữ liệu thường. Bộ injection hiện tại cũng chưa đủ khó để phân biệt hai prompt LLM. |
| `llm_models` | Model lớn có F1 cao hơn model nhỏ; LLM nhỉnh hơn Embedding + LogReg. | Hai LLM đều đạt F1 SUB = 0.974; Embedding + LogReg đạt 1.000. Model lớn tốn khoảng 9.9 lần chi phí API theo bảng đã làm tròn. | Model lớn không tự động tốt hơn trên mọi tập thử. SUB chỉ có 40 tin, cả hai LLM cùng sai một tin, nên không thể suy ra thứ hạng tổng quát từ chênh lệch này. |
| `llm_fewshot` | Thêm ví dụ sẽ tăng F1 và tăng chi phí khoảng 3 lần. | F1 vẫn là 0.974; chi phí tăng từ 0.009 lên 0.023 USD/1k tin, khoảng 2.6 lần. | Ví dụ làm prompt dài hơn nhưng chưa sửa được lỗi còn lại. Mô hình đã xử lý tốt phần lớn tin ngay từ zero-shot, nên thêm ví dụ chưa mang lại lợi ích trên SUB này. |

## 4. Phân tích 50 case khó

| Mô hình | F1 SUB thường | F1 50 case khó | Accuracy 50 case khó |
|---|---:|---:|---:|
| BoW + LogReg | 1.000 | 0.378 | 54% |
| BoW + MLP | 1.000 | 0.410 | 54% |
| MiniLM + kNN | 1.000 | 0.767 | 72% |
| MiniLM + LogReg | 1.000 | 0.640 | 64% |
| MiniLM + MLP head | 0.974 | 0.667 | 66% |
| LLM nhỏ zero-shot | 0.974 | 1.000 | 100% |
| LLM nhỏ + prompt phòng thủ | 0.974 | 1.000 | 100% |

**BoW + LogReg tụt nhiều nhất:** giảm 0.622 F1, tương đương 62.2 điểm phần trăm so với SUB; nếu so với F1 trên toàn bộ test là 0.943 thì giảm 56.5 điểm phần trăm. Mô hình sai 23/50 case, khó nhất ở nhóm thao túng/lừa đảo (đúng 1/8), né từ khóa (1/6) và spam giọng tự nhiên (2/10). BoW + MLP cũng yếu ở né từ khóa, đúng 0/6.

**Hai case thực sự xuất hiện trong danh sách lỗi của BoW + LogReg:**

1. **Case #21 — thao túng/lừa đảo; nhãn thật spam, dự đoán ham.**

   > Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account? I'll explain later x

   Tin nhắn giả người thân và yêu cầu chuyển tiền, nhưng dùng giọng trò chuyện bình thường. BoW chỉ tổng hợp token, không mô hình hóa đầy đủ quan hệ mạo danh và ý đồ xin tiền; các tín hiệu quen thuộc của spam trong tập train không còn rõ. Đây là cơ chế giải thích phù hợp với biểu diễn BoW, chưa phải phân tích đóng góp từng trọng số.

2. **Case #29 — né từ khóa; nhãn thật spam, dự đoán ham.**

   > C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m

   Các từ bị thay chữ bằng số nên không khớp với những token như `won` hoặc `claim` mà mô hình đã học. Tokenizer của lab chỉ tách token và chuyển chữ thường, chưa phục hồi kiểu viết này. Chuẩn hóa chữ/số có thể giúp, nhưng cần tránh sửa nhầm mã OTP, URL hoặc nội dung hợp lệ.

Embedding giảm phụ thuộc từ khóa nhưng vẫn nhầm khi ngữ nghĩa gần nhau mà nhãn khác nhau: kNN chỉ đúng 25% nhóm “nói về spam”. Hai LLM đúng toàn bộ 50 case, nên không có case tất cả mô hình cùng sai; prompt phòng thủ chưa tạo khác biệt về điểm số ở bất kỳ nhóm nào. Kết quả này chưa chứng minh LLM an toàn trước mọi prompt injection.

Bộ 50 case viết tay giúp tìm điểm yếu và kiểm tra phòng thủ, không đại diện cho tần suất spam thực tế. Notebook đã thêm 5 case bonus với sự hỗ trợ của AI, nhưng output đánh giá đang lưu vẫn là bộ 50 case; chưa có bằng chứng kết luận các case mới đánh lừa được LLM.

## 5. Thử thách cá nhân

**Đề #6**, do `8387 % 6 = 5` theo chỉ số bắt đầu từ 0: cho LLM trả JSON gồm `label` và `reason`, viết parser, so token/chi phí với bản trả một từ trên 10 tin SUB.

Tôi giữ hàm prompt một từ để làm đối chứng, thêm `build_classification_prompt_json` và `parse_label_json`. Hai cách dùng cùng model nhỏ, `temperature=0`, cùng 5 ham và 5 spam trong SUB. Giới hạn đầu ra là 32 token cho bản một từ và 160 token cho JSON; đây là giới hạn tối đa, không phải lượng token luôn tiêu thụ. Bản JSON dùng JSON mode; parser kiểm tra nhãn spam/ham và lý do không rỗng. Số token lấy từ `usage`, chi phí ước tính theo `PRICE_PER_1M` của lab.

| Cách trả lời | Accuracy | F1 | Input token | Output token | Tổng token | USD cho 10 tin |
|---|---:|---:|---:|---:|---:|---:|
| Một từ | 100% | 1.000 | 950 | 40 | 990 | 0.00011500 |
| JSON | 100% | 1.000 | 929 | 274 | 1203 | 0.00022990 |

JSON dùng tổng token gấp khoảng **1.22 lần**, chi phí gấp **2.00 lần**, nhưng F1 không tăng. Input JSON ít hơn một chút vì hai prompt có cách diễn đạt khác nhau; output tăng mạnh do cấu trúc JSON và câu giải thích. Với giá lab 0.10 USD/triệu token input và 0.50 USD/triệu token output, phần đầu ra làm chi phí tăng rõ hơn tổng số token.

Nếu chỉ cần lọc spam, bản một từ tiết kiệm hơn. JSON đáng dùng khi cần lưu lý do để kiểm tra và phân tích lỗi; giải thích được sinh ra không tự bảo đảm nhãn đúng. Kết luận này chỉ dựa trên 10 tin. Output mới nhất là 274 token đầu ra, khác con số 275 còn ghi trong phần diễn giải cũ của notebook.

## 6. Phương án cho sản phẩm thật

Yêu cầu 10 triệu SMS/ngày tương đương khoảng **116 tin/giây trung bình**, với **p95 < 50 ms/tin**.

Tôi chọn **MiniLM + kNN làm phương án nền để thử nghiệm sản phẩm** theo số liệu mới nhất: F1 test = 0.922, F1 SUB = 1.000, F1 case khó = 0.767, p95 = **19.227 ms/tin**, throughput cả lô khoảng **304 tin/giây**. Đây là mô hình có F1 cao nhất trên bộ khó trong nhóm đạt SLA; thời gian chuẩn bị khoảng 1.855 giây trong lần chạy lab. Tôi ưu tiên độ bền trên case khó thay vì chỉ chọn F1 test thường cao nhất của BoW + MLP (0.984).

Đánh đổi là cần lưu vector và nhãn, tìm láng giềng khi suy luận, và vẫn có nguy cơ chặn nhầm tin nói về spam hoặc tin hợp lệ. Kho mẫu tăng lớn sẽ làm tăng bộ nhớ và chi phí tìm kiếm. Throughput cả lô cao hơn 116 tin/giây chưa chứng minh p95 vẫn đạt khi chịu tải đồng thời hoặc lưu lượng giờ cao điểm; cần đo lại trên môi trường triển khai và dữ liệu đại diện hơn.

Có thể bổ sung router: dùng mức đồng thuận của k láng giềng và khoảng cách đã kiểm định trên validation để phát hiện tin khó, rồi chuyển một phần sang LLM nhỏ hoặc hậu kiểm. Cách này giảm số lần gọi API; tuy nhiên nhánh LLM có p95 khoảng 1.2–1.8 giây trong lab. Vì vậy, router đồng bộ phải được đo p95 toàn hệ thống, không thể mặc nhiên coi là đạt SLA.

**Ước tính nếu dùng LLM cho mọi tin trong 30 ngày**, theo cột USD/1k tin đã làm tròn ở Phần 5:

`Chi phí tháng = (10.000.000 / 1.000) × 30 × USD/1k tin`.

| Cách dùng LLM | USD/1k tin | USD/tháng |
|---|---:|---:|
| Nhỏ zero-shot | 0.009 | 2,700 |
| Lớn zero-shot | 0.089 | 26,700 |
| Nhỏ few-shot | 0.023 | 6,900 |

Đây là ước tính phí API theo bảng giá lab, chưa gồm hạ tầng. Bộ đo 40 tin ở Phần 4 ước lượng token bằng số từ; thử thách JSON dùng `usage` thực tế trên 10 tin khác mẫu. Nếu ngoại suy từ 10 tin đó, bản một từ khoảng **3,450 USD/tháng**, JSON khoảng **6,897 USD/tháng**. Hai cách ước lượng khác cơ sở nên cần đo `usage` trên tập đại diện trước khi lập ngân sách; không dùng con số chi phí này như cam kết vận hành.

## 7. AI log và điều đã học

Tôi hiểu rằng điểm rất cao trên tập test quen thuộc chưa bảo đảm mô hình xử lý được lừa đảo hoặc kiểu viết mới. Thêm độ phức tạp, model lớn, few-shot hay phần giải thích không tự động làm F1 tăng; phải đo lợi ích cùng chi phí và độ trễ. Tôi cũng học cách phân biệt số liệu đo thật với ước tính, và kiểm chứng nội dung AI bằng code cùng kết quả thực nghiệm.

