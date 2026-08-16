# Invoice_Scanner

Pipeline chính nằm toàn bộ trong `OCR_invoice_GGUF_local.ipynb`.

## Cách chạy

1. Mở notebook và chạy cell `Tải mmproj và mở server`. Cell này tải file
   `mmproj-Vintern-1B-v3_5-Q8_0.gguf` vào thư mục `models/` và in ra command
   server.

2. Mở terminal riêng, copy command được notebook in ra, ví dụ:

```powershell
llama-server -hf rootonchair/Vintern-1B-v3_5-GGUF-ext:Q6_K --mmproj "D:/Invoice_Scanner/models/mmproj-Vintern-1B-v3_5-Q8_0.gguf" --chat-template vicuna --port 8081
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

## Web app full-stack

Repo hiện có thêm app web gồm:

- `backend/`: FastAPI API nhận file, gọi pipeline Vintern GGUF local và trả JSON.
- `frontend/`: React 19 + TypeScript + Vite + React Router + Tailwind CSS + Axios.

### Cài đặt một lần

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r backend\requirements.txt

cd frontend
npm install
cd ..
```

### Chạy Vintern server

Giữ terminal này mở:

```powershell
llama-server -hf rootonchair/Vintern-1B-v3_5-GGUF-ext:Q4_K_M --mmproj "D:\Invoice_Scanner\models\mmproj-Vintern-1B-v3_5-Q8_0.gguf" --chat-template vicuna --port 8081
```

### Chạy backend

Mở terminal thứ hai:

```powershell
.\.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

API health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

### Chạy frontend

Mở terminal thứ ba:

```powershell
cd frontend
npm run dev -- --host 127.0.0.1
```

Mở app tại:

```text
http://127.0.0.1:5173/
```

Nếu backend hoặc frontend dùng port khác, chỉnh `frontend/.env` theo mẫu
`frontend/.env.example`.

## Chạy bằng Docker

Docker Compose chạy 2 container:

- `app`: backend FastAPI + frontend React đã build sẵn.
- `llama-server`: `llama.cpp` server chạy model Vintern GGUF.

Backend gọi OCR server qua network nội bộ Docker tại:

```text
http://llama-server:8081/v1
```

### Chuẩn bị

- Cài Docker Desktop và mở Docker Desktop trước khi chạy lệnh.
- Đảm bảo file `models/mmproj-Vintern-1B-v3_5-Q8_0.gguf` đã tồn tại.
- Lần chạy đầu cần internet để Docker tải image và `llama-server` tải model
  `rootonchair/Vintern-1B-v3_5-GGUF-ext:Q4_K_M` từ Hugging Face.

Kiểm tra file `mmproj`:

```powershell
Test-Path "D:\Invoice_Scanner\models\mmproj-Vintern-1B-v3_5-Q8_0.gguf"
```

Nếu trả `False`, chạy lại cell tải model trong notebook.

### 1. Build và chạy toàn bộ app

Mở terminal tại thư mục repo:

```powershell
cd D:\Invoice_Scanner
docker compose up --build
```

Lần đầu Docker sẽ:

- Tải image `ghcr.io/ggml-org/llama.cpp:server`.
- Build image web app.
- Cài dependency Python/Node.
- Build frontend React.
- Tải/cached model Vintern cho `llama-server`.

Bước tải model có thể mất vài phút. Khi thấy log Uvicorn chạy ở port `8000`,
mở app tại:

```text
http://localhost:8000/
```

API health check:

```powershell
Invoke-RestMethod http://localhost:8000/api/health
```

Kiểm tra riêng `llama-server`:

```powershell
Invoke-RestMethod http://localhost:8081/health
```

### 2. Chạy lại sau lần đầu

Nếu không đổi code:

```powershell
docker compose up
```

Nếu có đổi code frontend/backend hoặc đổi dependency:

```powershell
docker compose up --build
```

### 3. Dừng app

Nhấn `Ctrl+C` ở terminal đang chạy Docker, sau đó nếu muốn dừng hẳn container:

```powershell
docker compose down
```

Dữ liệu được lưu trong Docker volume:

- `invoice-storage`: file upload và kết quả JSON.
- `llama-cache`: cache model tải từ Hugging Face.

Nếu muốn xóa luôn dữ liệu và cache model:

```powershell
docker compose down -v
```

### Tùy chỉnh image llama.cpp

Mặc định dùng CPU image:

```text
ghcr.io/ggml-org/llama.cpp:server
```

Nếu muốn dùng image khác, ví dụ CUDA image, truyền `LLAMA_CPP_IMAGE` trước khi
chạy Docker.

PowerShell:

```powershell
$env:LLAMA_CPP_IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda"
docker compose up --build
```

macOS/Linux:

```bash
LLAMA_CPP_IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda" docker compose up --build
```

Lưu ý: CUDA cần Docker/NVIDIA runtime trên máy host. Nếu chưa cấu hình GPU cho
Docker, hãy dùng image CPU mặc định.

### Dùng OCR server bên ngoài thay vì container

Nếu muốn quay lại cách chạy `llama-server` bên ngoài Docker, truyền lại
`OCR_SERVER_URL`:

```powershell
$env:OCR_SERVER_URL="http://host.docker.internal:8081/v1"
docker compose up --build
```

### Lỗi thường gặp

Nếu app mở được nhưng OCR chưa chạy:

- Chờ thêm vài phút ở lần đầu, vì `llama-server` có thể vẫn đang tải model.
- Xem log model server:

```powershell
docker compose logs -f llama-server
```

- Xem log web app:

```powershell
docker compose logs -f app
```

Nếu port bị trùng, ví dụ máy đang có service khác dùng `8000` hoặc `8081`, đổi
port bên trái trong `docker-compose.yml`, ví dụ `"8001:8000"` hoặc `"8082:8081"`.
