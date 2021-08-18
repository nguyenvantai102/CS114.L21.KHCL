# TÓM TẮT BÁO CÁO
## THÀNH VIÊN NHÓM
| STT | Họ tên | MSSV | Email | Github |Lớp|
| :---: | --- | --- | --- | --- |---|
| 1 | Nguyễn Văn Tài | 19520250 | *19520250@gm.uit.edu.vn* | [nguyenvantai](https://github.com/nguyenvantai102) |CS114.L21.KHCL|
| 2 | Trần Xuân Nhơn | 18521212 | *18521212@gm.uit.edu.vn* | [18521212](https://github.com/18521212)|CS114.L21.KHCL|
| 3 | Nguyễn Ngọc Trưởng | 19522440 | *19522440@gm.uit.edu.vn* |[nguyenngoctruong2k1](https://github.com/nguyenngoctruong)|CS114.L22.KHCL|

## 1.	BÀI TOÁN
### a. Bài toán
- Đối với tình trạng giao thông Việt Nam ngày càng phát triển dẫn đến các vấn đề như ùn tắc giao thông ngày càng trở nên phổ biến. Việc xây dựng một hệ thống có thể giúp cải thiện tình hình giao thông là một vấn đề rất được quan tâm ngày nay. Dựa trên vấn đề đó, nhóm em thực hiện xây dựng một hệ thống đếm lưu lượng phương tiện cơ bản dựa trên thuật toán Deep Learning. 
- Hệ thống của nhóm em sử dụng hai modun chính: Modun nhận diện loại phương tiện (YOLOv4) và modun đếm số lượng phương tiện (OpenCV). Chỉ số mAP của thuật toán YOLOv4 là 83.56% trên bộ dataset của nhóm thu thập. Ban đầu, nhóm đã sử dụng YOLOv3 để tiến hành nhận diện loại phương tiện, tuy nhiên YOLOv3 cho kết quả không được cao (chỉ số mAP khoảng 78.1%), do đó nhóm đã ưu tiên sử dụng YOLOv4 để đạt được độ chính xác cao hơn.
### b. Dataset
- Bộ dữ liệu của nhóm là hình ảnh được thu thập từ camera của Sở giao thông thành phố Hồ Chí Minh. Mỗi camera cho chất lượng ảnh không giống nhau, bên cạnh đó tùy vào từng thời điểm mà số lượng phương tiện có sự khác nhau rất lớn (ví dụ, hiện tại buổi tối không có hoặc rất ít phương tiện đi trên đường). Do đó nhóm phải tiến hành chọn lọc những hình ảnh phù hợp với yêu cầu ngữ cảnh bài toán, không crawl ảnh một cách không chọn lọc tránh ảnh hưởng đến độ chính xác của hệ thống. Nhóm tiến hành xây dựng bộ dữ liệu với 5 class: motorbike, bicycle, van, truck và car. Sử dụng công cụ LabelImg để tiến hành gán nhãn các đối tượng có trong ảnh.
- Dữ liệu ban đầu của nhóm thu thập được là 1668 ảnh. Nhóm thu thập dữ liệu ít như vậy bỡi vì nhóm thống kê được 1 vài phương tiện đủ lớn (gần bằng 2000). Do vậy nhóm tiến hành training rồi đánh giá model thử xem như thế nào vào có phương án cải thiện model.
- Sau khi đã training thử với mô hình YOLOv3 và v4 thì nhóm thấy chỉ số mAP còn thấp. Để tiến hành cải thiện hệ thống, nhóm đã tăng cường thêm 454 bức ảnh.
  Vì tăng cường dữ liệu tập trung chủ yếu vào các đối tượng có chỉ số mAP thấp, cần phải chọn lọc và đợi đến frame ảnh hợp lý theo yêu cầu. Vì một vài lý do chủ quan về tình hình dịch COVID nên do vậy chiếm nhiều thời gian, tuy nhiên thì vẫn không khả quan nhiều.
## 2.	CẬP NHẬP THAY ĐỔI
Với những góp ý từ thầy An trong buổi báo cáo, nhóm đã tiến hành những cập nhật cho bài báo cáo như sau:
- Tìm kiếm các bài báo nghiên cứu của Việt Nam về lĩnh vực đề tài của nhóm với bộ dữ liệu ở thành phố Hồ Chí Minh.
- Thêm những ví dụ hình ảnh cụ thể để giải thích các vấn đề
## 3. CÁC FILE BÁO CÁO
- [File báo cáo chính](https://github.com/nguyenvantai102/CS114.L21.KHCL/blob/main/FinalProject/File%20B%C3%A1o%20C%C3%A1o.pdf)
- [File Colab](https://github.com/nguyenvantai102/CS114.L21.KHCL/blob/main/FinalProject/FinalProject_(B%E1%BA%A3n_%C4%91%E1%BA%A7y_%C4%91%E1%BB%A7).ipynb)
- [Source Code demo](https://github.com/nguyenvantai102/CS114.L21.KHCL/blob/main/FinalProject/DemoApplication.py)
- [Tập dữ liệu](https://github.com/nguyenvantai102/CS114.L21.KHCL/tree/main/FinalProject/Dataset)
