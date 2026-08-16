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
llama-server -hf rootonchair/Vintern-1B-v3_5-GGUF-ext:Q4_K_M --mmproj "D:\Invoice_Scanner\models\mmproj-Vintern-1B-v3_5-Q8_0.gguf" --chat-template vicuna --n-gpu-layers 0 --port 8081
```

Nếu `llama-server` của bạn là bản có CUDA và máy có NVIDIA GPU, đổi
`--n-gpu-layers 0` thành `--n-gpu-layers 999` để offload model lên GPU.

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
.\scripts\docker-up.ps1
```

Script này tự kiểm tra `nvidia-smi`. Nếu máy có NVIDIA GPU và Docker hỗ trợ GPU,
nó sẽ dùng `docker-compose.gpu.yml`, CUDA image và `LLAMA_N_GPU_LAYERS=999`.
Nếu không có GPU, nó tự chạy CPU như cũ.

Lần đầu Docker sẽ:

- Tải image `ghcr.io/ggml-org/llama.cpp:server` hoặc CUDA image nếu bật GPU.
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
.\scripts\docker-up.ps1 -NoBuild
```

Nếu có đổi code frontend/backend hoặc đổi dependency:

```powershell
.\scripts\docker-up.ps1
```

Chạy nền:

```powershell
.\scripts\docker-up.ps1 -Detached
```

Ép chạy CPU dù máy có GPU:

```powershell
.\scripts\docker-up.ps1 -Cpu
```

### Tăng tốc bằng GPU

Docker GPU mode hiện hỗ trợ NVIDIA/CUDA. Máy cần:

- NVIDIA driver đang hoạt động (`nvidia-smi` chạy được trong PowerShell).
- Docker Desktop bật WSL2 backend và hỗ trợ GPU container.

Cách chạy thủ công không dùng script:

```powershell
$env:LLAMA_N_GPU_LAYERS="999"
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Kiểm tra log xem model có offload lên GPU không:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f llama-server
```

### Chế độ OCR PDF

Mặc định Docker dùng:

```text
OCR_PDF_TEXT_STRATEGY=assist
```

Ở chế độ này, PDF có text layer vẫn được render thành ảnh và chạy qua Vintern
vision model. Text layer chỉ được dùng làm lớp hỗ trợ để sửa các trường dễ sai
như số hóa đơn, mã số thuế, dòng hàng và tổng tiền.

Các giá trị có thể dùng:

- `assist`: chạy vision model cho trang có nội dung, rồi dùng text layer hỗ trợ.
- `fast`: nếu text layer đủ tin cậy thì bỏ qua vision model để trả kết quả nhanh.
- `off`: không dùng text layer, chỉ dùng vision model.

Ví dụ muốn benchmark chế độ chỉ dùng model:

```powershell
$env:OCR_PDF_TEXT_STRATEGY="off"
docker compose up --build
```

Ví dụ muốn chạy nhanh với PDF điện tử:

```powershell
$env:OCR_PDF_TEXT_STRATEGY="fast"
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

### Tùy chỉnh image llama.cpp/GPU

Mặc định dùng CPU image:

```text
ghcr.io/ggml-org/llama.cpp:server
```

Khi dùng `.\scripts\docker-up.ps1`, script sẽ tự đổi sang CUDA image nếu phát
hiện NVIDIA GPU. Nếu muốn tự chọn image khác, truyền `LLAMA_CPP_IMAGE` trước khi
chạy script:

PowerShell:

```powershell
$env:LLAMA_CPP_IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda"
$env:LLAMA_N_GPU_LAYERS="999"
.\scripts\docker-up.ps1
```

Hoặc chạy thủ công bằng compose override:

```powershell
$env:LLAMA_CPP_IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda"
$env:LLAMA_N_GPU_LAYERS="999"
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Lưu ý: CUDA cần Docker/NVIDIA runtime trên máy host. Nếu máy dùng AMD/Intel GPU
hoặc Docker chưa thấy GPU, hãy dùng CPU mode hoặc cài image/runtime phù hợp rồi
trỏ lại `LLAMA_CPP_IMAGE`.

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
