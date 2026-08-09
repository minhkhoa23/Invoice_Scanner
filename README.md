# Invoice_Scanner

Pipeline chính nằm toàn bộ trong `OCR_invoice_GGUF_local.ipynb`.

## Cách chạy

1. Mở notebook và chạy cell `Tải mmproj và mở server`. Cell này tải file
   `mmproj-Vintern-1B-v3_5-Q8_0.gguf` vào thư mục `models/` và in ra command
   server.

2. Mở terminal riêng, copy command được notebook in ra, ví dụ:

```powershell
llama-server -hf rootonchair/Vintern-1B-v3_5-GGUF-ext:Q4_K_M --mmproj "D:/Invoice_Scanner/models/mmproj-Vintern-1B-v3_5-Q8_0.gguf" --chat-template vicuna --port 8081
```

Giữ terminal server đó mở.

Nếu server báo không tìm thấy `mmproj`, kiểm tra file:

```powershell
Test-Path "D:\Invoice_Scanner\models\mmproj-Vintern-1B-v3_5-Q8_0.gguf"
```

Nếu trả `False`, chạy lại cell `Tải mmproj và mở server` trong notebook.

3. Đổi biến `INPUT_PATH` thành ảnh hoặc PDF:

```python
INPUT_PATH = Path("D:/Invoice_Scanner/invoice.pdf")
```

4. Chạy notebook từ trên xuống.

Notebook sẽ:

- Nhận input là ảnh hoặc PDF.
- Nếu là PDF, render từng trang thành ảnh trong thư mục `*_images`.
- Gửi ảnh vào Vintern GGUF qua `llama.cpp server`.
- Lưu output JSON theo schema demo vào file `.vintern_gguf.json`.
