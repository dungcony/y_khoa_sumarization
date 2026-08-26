# Tóm tắt văn bản y khoa tiếng Việt

Ứng dụng Streamlit sử dụng mô hình `VietAI/vit5-base` đã fine-tune trên tập tin tức y khoa tiếng Việt. Người dùng nhập một bài viết, điều chỉnh giới hạn độ dài và số beam, sau đó nhận bản tóm tắt cùng tỷ lệ nén và thời gian suy luận.

## 1. Yêu cầu môi trường

- Python 3.10 trở lên.
- Khuyến nghị tối thiểu 8 GB RAM và 3 GB dung lượng trống.
- GPU CUDA không bắt buộc; ứng dụng tự chuyển sang CPU nếu không có GPU.
- Kết nối Internet chỉ cần khi cài dependency hoặc tải mô hình lần đầu.

## 2. Chuẩn bị mô hình

Mô hình fine-tune được phát hành công khai tại:

<https://huggingface.co/dungcony/vit5-base-y-khoa>

Người dùng không cần tải hoặc chép thủ công thư mục `output`. Ở lần chạy đầu,
ứng dụng tự tạo thư mục `model/vit5-base-y-khoa/`, tải khoảng 905 MB tệp mô hình
từ Hugging Face vào đó và sử dụng lại ở những lần chạy sau. Thư mục này được bỏ
qua bởi Git nên trọng số không bị đưa lên GitHub.

Có thể chỉ định một thư mục local hoặc model ID khác bằng biến môi trường
`VIT5_MODEL_SOURCE`. Khi biến này được đặt, ứng dụng dùng trực tiếp nguồn được
chỉ định thay cho thư mục mặc định.

## 3. Cài đặt và chạy nhanh

### Linux hoặc macOS

```bash
git clone https://github.com/dungcony/y_khoa_sumarization.git
cd y_khoa_sumarization
chmod +x run.sh
./run.sh
```

### Windows

Tải hoặc clone repository, sau đó mở Command Prompt ngay tại thư mục gốc
`y_khoa_sumarization` và chạy:

```bat
run.bat
```

Hai script trên tự tạo `.venv`, cài dependency từ `pyproject.toml` và khởi chạy Streamlit.

## 4. Cài đặt thủ công

```bash
git clone https://github.com/dungcony/y_khoa_sumarization.git
cd y_khoa_sumarization
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m streamlit run app.py
```

Trên Windows, lệnh kích hoạt môi trường là:

```bat
.venv\Scripts\activate
```

Sau khi máy chủ khởi động, mở `http://localhost:8501` trong trình duyệt.

## 5. Sử dụng ứng dụng

1. Dán bài báo hoặc văn bản y khoa vào ô **Văn bản gốc**.
2. Giữ cấu hình mặc định `160` token và `2` beam để đồng bộ với phép đánh giá trong báo cáo.
3. Nhấn **Tóm Tắt Ngay**.
4. Kiểm tra bản tóm tắt, tỷ lệ nén và thời gian suy luận hiển thị ở cột kết quả.

Đầu vào được thêm tiền tố `summarize:` và cắt tại 768 token. Kết quả có tính xác định vì ứng dụng không lấy mẫu ngẫu nhiên và chặn lặp 3-gram.

## 6. Lỗi thường gặp

### Không tìm thấy mô hình

Kiểm tra kết nối Internet và truy cập
<https://huggingface.co/dungcony/vit5-base-y-khoa>. Nếu dùng mô hình local, cần
có đủ `model.safetensors`, tokenizer và các tệp cấu hình trong cùng một thư mục.
Nếu lần tải đầu bị gián đoạn, chạy lại ứng dụng để tiếp tục tải; khi thư mục tải
dở không thể phục hồi, xóa `model/vit5-base-y-khoa/` rồi chạy lại.

### Máy thiếu bộ nhớ

Đóng các tiến trình Python khác và giảm số beam về `1`. CPU vẫn chạy được nhưng thời gian suy luận dài hơn GPU.

### Cổng 8501 đã được sử dụng

```bash
python -m streamlit run app.py --server.port=8502
```

### Tải lại mô hình

```bash
rm -rf model/vit5-base-y-khoa
python -m streamlit run app.py
```

Trên Windows, xóa thư mục `model\vit5-base-y-khoa` bằng File Explorer rồi chạy
lại `run.bat`.

## 7. Nguồn mã

Kho mã của đề tài: <https://github.com/dungcony/y_khoa_sumarization>
