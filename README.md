# RNN Product Recommender

## Chạy local

1. Mở và chạy toàn bộ `RNN_ProductRecommendation.ipynb`. Cell cuối của phần huấn luyện sẽ sinh `models/rnn_product_recommender.keras`, `artifacts.pkl`, `catalog.json` và `metrics.json`.
2. Cài môi trường: `pip install -r requirements.txt`.
3. Chạy: `python app.py`, sau đó mở `http://localhost:5000`.

## Deploy thủ công lên Render

1. Đưa riêng thư mục `RNN_GoiYSanPham` (bao gồm cả thư mục `models/` sau huấn luyện) lên một GitHub repository.
2. Trong Render: **New +** → **Web Service** → kết nối repository. Nếu repository có cấp cha, đặt **Root Directory** là `RNN_GoiYSanPham`.
3. Repository đã có `.python-version` để Render dùng Python 3.11, tương thích TensorFlow. Nếu service đã được tạo trước đó, vào **Environment** và đặt `PYTHON_VERSION` thành `3.11.11`, rồi lưu thay đổi.
4. Đặt Build Command là `pip install -r requirements.txt` và Start Command là `gunicorn app:app`, sau đó deploy. Render tự cung cấp biến môi trường `PORT`; không cần cấu hình thêm.

> Lưu ý: file model phải được commit lên repository hoặc lấy từ persistent disk/object storage. Render free instance có filesystem tạm thời, vì vậy không chạy notebook để train ngay trên web service.
