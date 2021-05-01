
# ĐỀ BÀI:
- Mỗi nhóm tìm dăm ba ví dụ về bài toán regression ***TRONG THỰC TẾ***
- Ghi rõ input, output và cách thu thập + xử lý data, commit vào github repository và dẫn link lên Google Classroom.

## Thành viên:
- Nguyễn Văn Tài - 19520250 - 19520250@gm.uit.edu.vn
- Trần Xuân Nhơn - 18521212 - 18521212@gm.uit.edu.vn

## Bài toán Regression trong thực tế

### Bài toán 1: Dự đoán giá tiền đi Grab
- Input: Khoảng cách người dùng đi (km), điều kiện thời tiết (mưa và không mưa), mật độ giao thông(thông thoáng, tắt đường).
- Output: Số tiền phải trả cho chuyến đi
- Cách thu thập data:
  - Thu thập dữ liệu từ người dùng đã đi (số km, thời tiết, điều kiện giao thông và giá tiền)
- Cách xử lý data:
  - Xây dựng một file csv với 4 cột là số km(số thực), thời tiết (0 và 1), giao thông (0 và 1) và giá tiền.
  - Chuẩn hóa dữ liệu về dạng hợp với model

### Bài toán 2: Dự đoán thời gian có số lượng khách hàng đến mua hàng theo yêu cầu
-	Input: Tập thông tin hóa đơn mua hàng theo ngày, giờ của một trung tâm thương mại.
-	Output: Dự đoán thời gian có số khách hàng đến mua hàng theo yêu cầu.
-	Cách thu thập: Lấy dữ liệu từ các hóa đơn mua hàng.
-	Xử lý data:
  - Đồ thị có hai cột: cột thời gian(ngày) và cột số khách hàng mua hàng. 
    -Số khách hàng mua hàng lấy từ thông tin hóa đơn, các hóa đơn có cùng mã khách hàng và cùng thời gian mua hàng chỉ lấy một lần. 
    - Đếm số mã khách hàng tìm ra được số khách hàng trong từng thời điểm. 
    
### Bài toán 3: Dự đoán trường hợp có tác dụng phụ sau khi tiêm vacine COVID-19 của một quốc gia
-	Input: Thông tin tiêm Vacine của một quốc gia gồm mã bệnh nhân, độ tuổi.
-	Output: Tính tỷ lệ (một số thực a, 0<=a<=100) có tác dụng phụ sau khi tiêm vacine COVID-19
-	Cách thu thập: Lấy dữ liệu từ bộ y tế gồm mã bệnh nhân, thông tin tình trạng y tế của bệnh nhân.
-	Xử lý data: Gom dữ liệu thành file csv:
  - Xử lý những hàng không hợp lệ
  - Chuyển đổi dữ liệu thành dạng dữ liệu phù hợp với model.

### Bài toán 4: Dự đoán điểm thi cuối kì của một môn học
- Tham khảo: [http://tutorials.aiclub.cs.uit.edu.vn/index.php/2021/04/24/linear-regression/?fbclid=IwAR3z50DDHvqRg8vJZACvhBDYdd0_yYwYs6fGDEu_Rf6IrGxY_1h4ZrFnrtY]
- Input: Điểm thi giữa kỳ (kiểu số nguyên)
- Output: Điểm dự đoán cuối kì
- Thu thập dữ liệu:
  - Điểm từ giáo viên
  - Khảo sát từ những sinh viên
- Xử lý dữ liệu: Gom dữ liệu thành file csv:
  - Chuẩn hóa dữ liệu về dạng phù hợp với model
      
