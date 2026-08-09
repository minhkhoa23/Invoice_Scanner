# Invoice_Scanner

## Pipeline GGUF local CPU

Pipeline mới nằm ở `ocr_invoice_gguf_local.py` và notebook ví dụ nằm ở
`OCR_invoice_GGUF_local.ipynb`.

Cài dependency:

```bash
pip install --prefer-binary -r requirements-gguf-local.txt
```

`--input` nhận được cả ảnh và PDF. Nếu input là PDF, option
`--save-pdf-images` sẽ render từng trang thành ảnh trong thư mục
`<ten-file-pdf>_images` trước khi OCR.

Chạy với PDF mẫu:

```bash
python ocr_invoice_gguf_local.py --input "4f30c1ad-c130-47a0-b164-41ad070218f2.pdf" --output "invoice.gguf.json" --save-pdf-images
```

Chạy với ảnh:

```bash
python ocr_invoice_gguf_local.py --input "invoice.jpg" --output "invoice.gguf.json"
```

Mặc định pipeline dùng `rootonchair/Vintern-1B-v3_5-GGUF-ext` quant `Q4_K_M`
và ép `n_gpu_layers=0` để chạy CPU-only. Có thể đổi sang `Q4_K_S` hoặc
`IQ4_XS` nếu muốn nhẹ hơn:

```bash
python ocr_invoice_gguf_local.py --input "invoice.pdf" --quant Q4_K_S
```

Nếu Windows vẫn tải file `.tar.gz` và build `llama-cpp-python` từ source,
hãy cài riêng wheel CPU dựng sẵn:

```bash
pip install --prefer-binary --only-binary llama-cpp-python llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```
