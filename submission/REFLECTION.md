# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**  
Điều làm tôi ngạc nhiên nhất là việc so sánh cùng số tham số giữa `attn_only` ($r=283$) và `correct` ($r=16$, all text-linear) cho thấy `attn_only` có loss huấn luyện thấp hơn nhưng điểm kiểm thử target lại kém hơn. Tôi từng nghĩ rank càng cao thì mô hình càng mạnh, nhưng thực tế việc mở rộng vị trí gắn adapter đến mọi khối linear của text decoder mới là yếu tố quyết định khả năng tổng quát hóa.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**  
Tôi mất nhiều thời gian nhất ở khâu kiểm chứng loss mask và chat template (NB1). Trước khi làm, tôi dự đoán huấn luyện (NB3/NB4) sẽ tốn nhiều thời gian nhất, nhưng việc kiểm tra từng span ký tự và giải mã ngược token để chắc chắn prompt không bị tính loss mới là khâu đòi hỏi sự tỉ mỉ và cẩn trọng cao nhất.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**  
Trước lab này, tôi từng tin rằng "cứ fine-tune là sẽ tốt hơn prompting" và "đánh giá fine-tune thì chỉ cần theo dõi perplexity hay training loss giảm là xong". Sau lab, tôi hiểu rằng một prompt được kỹ thuật hóa tử tế (baseline b) là một rào cản rất lớn, và chỉ có đánh giá trực tiếp trên năng lực tác vụ đích cùng cổng hồi quy mới chứng minh được giá trị thực của bản fine-tune.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**  
Tôi dùng AI assistant để hỗ trợ phân tích traceback lỗi, kiểm tra đối chiếu công thức tham số LoRA và rà soát cấu trúc báo cáo. Chỗ AI dễ sai nhất là xu hướng mặc định khuyên dùng `bf16=True` trên mọi GPU (kể cả chip Turing như T4 vốn không hỗ trợ phần cứng cho bf16) hoặc đề xuất tăng rank thay vì kiểm tra lại Learning Rate.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**  
Bước đầu tiên tôi làm không phải là tải model về train ngay, mà là đóng băng một tập dữ liệu đánh giá đại diện, xây dựng một baseline prompt tối ưu (baseline b), và đo đạc cẩn thận xem liệu prompting đã giải quyết được yêu cầu nghiệp vụ chưa. Chỉ khi prompt tối ưu chưa đạt ngưỡng hoặc chi phí/độ trễ context quá lớn, tôi mới bắt tay thiết lập pipeline fine-tune với loss mask được kiểm chứng nghiêm ngặt.
