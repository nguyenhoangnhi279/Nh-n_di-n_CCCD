# 🔍 Hệ Thống Trích Xuất Thông Tin Căn cước công dân
Dự án ứng dụng Deep Learning để tự động nhận diện và trích xuất thông tin văn bản từ hình ảnh thẻ Căn cước công dân. Hệ thống kết hợp giữa mô hình phát hiện vật thể và mô hình nhận dạng chữ viết tay/in ấn tiếng Việt.

## Công Nghệ Sử Dụng
* **YOLOv9 (Ultralytics):** Huấn luyện tùy chỉnh (Fine-tuning) để phát hiện và khoanh vùng (bounding box) các trường thông tin trên thẻ (Ví dụ: Số ID, Họ tên, Ngày sinh, Địa chỉ).
* **VietOCR:** Mô hình kiến trúc VGG-Seq2Seq chuyên dụng nhận dạng ký tự quang học (OCR) cho tiếng Việt, có khả năng xử lý tốt văn bản có dấu.
* **OpenCV & Pillow:** Tiền xử lý, cắt ảnh và vẽ khung hiển thị.
* **Gradio:** Xây dựng giao diện Web UI tương tác trực quan.

## Yêu Cầu
Khuyến nghị sử dụng **Python 3.10** hoặc **3.11** để tránh các lỗi xung đột thư viện với `vietocr`.

Cài đặt các thư viện cần thiết:
```bash
pip install ultralytics vietocr opencv-python gradio matplotlib pillow numpy
```
## Hướng Dẫn Cài Đặt & Khởi Chạy
Bước 1: Clone hoặc tải mã nguồn dự án về máy tính.

Bước 2: Đảm bảo file trọng số đã huấn luyện của YOLOv9 (đặt tên là best.pt) nằm cùng thư mục gốc với file app.py.

Bước 3: Khởi chạy ứng dụng bằng lệnh:
```bash
python app.py
```
Bước 4: Mở trình duyệt web và truy cập vào đường dẫn cục bộ được cung cấp ở terminal.