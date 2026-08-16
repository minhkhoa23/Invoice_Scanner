"""
Local CPU invoice OCR pipeline using the quantized GGUF Vintern model.

Examples:
    python ocr_invoice_gguf_local.py --input "invoice.pdf" --output "invoice.json"
    python ocr_invoice_gguf_local.py --input "invoice.jpg" --output "invoice.json"

The default CLI backend uses llama.cpp server, which is more reliable on
Windows than depending on llama-cpp-python multimodal wheels:
    llama-server -hf rootonchair/Vintern-1B-v3_5-GGUF-ext:Q4_K_M
    python ocr_invoice_gguf_local.py --backend server --input "invoice.pdf"
"""

from __future__ import annotations

import argparse
import ast
import base64
import importlib.metadata
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any


MODEL_REPO_ID = "rootonchair/Vintern-1B-v3_5-GGUF-ext"
MIN_LLAMA_CPP_VERSION = "0.3.10"
LLAMA_CPP_UPGRADE_COMMAND = (
    "pip install --upgrade --force-reinstall --prefer-binary "
    "--only-binary llama-cpp-python "
    f"\"llama-cpp-python>={MIN_LLAMA_CPP_VERSION}\" "
    "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
)

QUANT_FILES = {
    "Q2_K": "Vintern-1B-v3_5-Q2_K.gguf",
    "Q3_K_S": "Vintern-1B-v3_5-Q3_K_S.gguf",
    "Q3_K_M": "Vintern-1B-v3_5-Q3_K_M.gguf",
    "Q3_K_L": "Vintern-1B-v3_5-Q3_K_L.gguf",
    "IQ4_XS": "Vintern-1B-v3_5-IQ4_XS.gguf",
    "Q4_K_S": "Vintern-1B-v3_5-Q4_K_S.gguf",
    "Q4_K_M": "Vintern-1B-v3_5-Q4_K_M.gguf",
    "Q5_K_S": "Vintern-1B-v3_5-Q5_K_S.gguf",
    "Q5_K_M": "Vintern-1B-v3_5-Q5_K_M.gguf",
    "Q6_K": "Vintern-1B-v3_5-Q6_K.gguf",
    "F16": "Vintern-1B-v3_5-f16.gguf",
}

IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def require_pillow() -> tuple[Any, Any]:
    try:
        from PIL import Image, ImageOps
    except ImportError as error:  # pragma: no cover - helper message for users
        raise SystemExit(
            "Missing dependency: pillow. Install with:\n"
            "  pip install -r requirements-gguf-local.txt"
        ) from error

    return Image, ImageOps


def require_pymupdf() -> Any:
    try:
        import fitz
    except ImportError as error:  # pragma: no cover - helper message for users
        raise SystemExit(
            "Missing dependency: pymupdf. Install with:\n"
            "  pip install -r requirements-gguf-local.txt"
        ) from error

    return fitz


def require_requests() -> Any:
    try:
        import requests
    except ImportError as error:  # pragma: no cover - helper message for users
        raise SystemExit(
            "Missing dependency: requests. Install with:\n"
            "  pip install -r requirements-gguf-local.txt"
        ) from error

    return requests


def get_llama_cpp_diagnostics() -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "version": None,
        "chat_formats": [],
    }

    try:
        diagnostics["version"] = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        diagnostics["version"] = None

    try:
        from llama_cpp import llama_chat_format
    except ImportError:
        return diagnostics

    for name in dir(llama_chat_format):
        value = getattr(llama_chat_format, name)
        handlers = getattr(value, "_chat_handlers", None)
        if isinstance(handlers, dict):
            diagnostics["chat_formats"] = sorted(handlers)
            break

    return diagnostics


def require_chat_format(chat_format: str | None) -> None:
    if not chat_format:
        return

    diagnostics = get_llama_cpp_diagnostics()
    chat_formats = diagnostics["chat_formats"]

    if chat_formats and chat_format not in chat_formats:
        raise RuntimeError(
            "Installed llama-cpp-python does not support "
            f"chat_format='{chat_format}'.\n"
            f"Current version: {diagnostics['version']}\n"
            f"Available chat formats: {chat_formats}\n\n"
            "This GGUF vision pipeline needs the mtmd multimodal handler. "
            "Run this in the same notebook/kernel environment, then restart "
            "the kernel and reload the model:\n"
            f"  {LLAMA_CPP_UPGRADE_COMMAND}"
        )


INVOICE_JSON_SCHEMA = {
    "invoice": {
        "invoice_type": "VAT_INVOICE | SALES_INVOICE | RECEIPT | OTHER | null",
        "invoice_number": "string | null",
        "series": "string | null",
        "invoice_date": "YYYY-MM-DD | null",
        "tax_authority_code": "string | null",
        "currency": "string | null",
        "payment_method": "string | null",
    },
    "seller": {
        "name": "string | null",
        "english_name": "string | null",
        "tax_code": "string | null",
        "address": "string | null",
        "phone": ["string"],
        "fax": ["string"],
        "email": "string | null",
        "website": "string | null",
    },
    "buyer": {
        "name": "string | null",
        "company_name": "string | null",
        "tax_code": "string | null",
        "address": "string | null",
        "id_card": "string | null",
        "passport_number": "string | null",
        "account_number": "string | null",
        "budgetary_unit_code": "string | null",
    },
    "shipping": {
        "ship_from_warehouse": "string | null",
    },
    "items": [
        {
            "line_number": "number | null",
            "description": "string | null",
            "description_lines": ["string"],
            "item_type": "GOODS | SERVICE | FEE | OTHER | null",
            "container_number": "string | null",
            "unit": "string | null",
            "quantity": "number | null",
            "unit_price": "number | null",
            "amount": "number | null",
            "taxable_amount": "number | null",
            "vat_rate": "number | null",
            "vat_amount": "number | null",
            "total_amount": "number | null",
        }
    ],
    "totals": {
        "subtotal": "number | null",
        "vat_rate": "number | null",
        "vat_amount": "number | null",
        "total_payment": "number | null",
        "amount_in_words": "string | null",
    },
    "signature": {
        "is_valid": "boolean | null",
        "signed_by": "string | null",
        "signed_date": "YYYY-MM-DD | null",
    },
    "metadata": {
        "source_type": "PDF | IMAGE | null",
        "page_count": "number | null",
        "ocr_processed": True,
    },
}


INVOICE_PROMPT = f"""
Bạn là hệ thống OCR và trích xuất thông tin từ hóa đơn Việt Nam.

Nhiệm vụ:
- Đọc toàn bộ nội dung trong ảnh hóa đơn.
- Trả về duy nhất một JSON object hợp lệ, không dùng markdown, không giải thích.
- Nếu không thấy trường nào thì để null. Nếu không có danh sách thì dùng [].
- Chuẩn hóa ngày về YYYY-MM-DD nếu có thể.
- Với số tiền, số lượng, đơn giá: trả về number, bỏ dấu phân cách hàng nghìn.
- Giữ nguyên tiếng Việt có dấu trong tên công ty, địa chỉ và mô tả hàng hóa.
- Với bảng hàng hóa/dịch vụ, không bỏ dòng mô tả bị xuống dòng. Gộp vào description và lưu từng dòng gốc trong description_lines.
- Nếu mô tả có mã container như TGBU8540187 thì lưu vào container_number, vẫn giữ nguyên trong description.
- Với từng item, amount/taxable_amount là thành tiền trước thuế; vat_rate, vat_amount và total_amount là thuế suất, tiền thuế và tổng tiền thanh toán của riêng dòng đó nếu có.
- Không tự bịa thông tin không có trên ảnh.

Schema bắt buộc:
{json.dumps(INVOICE_JSON_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()


def get_quant_filename(quant: str | None = None, filename: str | None = None) -> str:
    if filename:
        return filename

    quant_key = (quant or "Q4_K_M").upper()

    if quant_key not in QUANT_FILES:
        choices = ", ".join(sorted(QUANT_FILES))
        raise ValueError(f"Unknown quant '{quant}'. Choose one of: {choices}")

    return QUANT_FILES[quant_key]


def load_llm(
    repo_id: str = MODEL_REPO_ID,
    quant: str = "Q4_K_M",
    filename: str | None = None,
    n_ctx: int = 8192,
    n_threads: int | None = None,
    chat_format: str | None = "mtmd",
    verbose: bool = False,
) -> Any:
    """
    Load the GGUF model through llama-cpp-python.

    n_gpu_layers=0 keeps inference on CPU for machines without a discrete GPU.
    """
    try:
        from llama_cpp import Llama
    except ImportError as error:  # pragma: no cover - helper message for users
        raise SystemExit(
            "Missing dependency: llama-cpp-python. Install with:\n"
            "  pip install -r requirements-llama-cpp-python.txt"
        ) from error

    model_filename = get_quant_filename(quant=quant, filename=filename)

    load_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "filename": model_filename,
        "n_ctx": n_ctx,
        "n_threads": n_threads or max(1, (os.cpu_count() or 4) - 1),
        "n_gpu_layers": 0,
        "verbose": verbose,
    }

    if chat_format:
        require_chat_format(chat_format)
        load_kwargs["chat_format"] = chat_format

    try:
        return Llama.from_pretrained(**load_kwargs)
    except Exception as error:
        if chat_format == "mtmd":
            raise SystemExit(
                "Could not load llama-cpp-python with chat_format='mtmd'. "
                "This model needs the mtmd multimodal chat handler for images.\n"
                "Try upgrading llama-cpp-python and restarting the kernel:\n"
                f"  {LLAMA_CPP_UPGRADE_COMMAND}"
            ) from error

        raise


def normalize_image(
    image: Any,
    max_side: int = 1800,
) -> Any:
    Image, ImageOps = require_pillow()

    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def image_to_data_uri(
    image: Any,
    max_side: int = 1800,
    jpeg_quality: int = 92,
) -> str:
    image = normalize_image(image=image, max_side=max_side)

    buffer = BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=jpeg_quality,
        optimize=True,
    )

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def pdf_to_images(
    pdf_path: str | Path,
    dpi: int = 200,
) -> list[Any]:
    fitz = require_pymupdf()
    Image, _ = require_pillow()

    document = fitz.open(pdf_path)
    images: list[Any] = []

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_number in range(len(document)):
        page = document.load_page(page_number)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes(
            "RGB",
            (pixmap.width, pixmap.height),
            pixmap.samples,
        )
        images.append(image)

    document.close()
    return images


def default_pdf_image_dir(pdf_path: str | Path) -> Path:
    path = Path(pdf_path)
    return path.with_name(f"{path.stem}_images")


def save_images_to_files(
    images: list[Any],
    output_dir: str | Path,
    stem: str = "page",
    image_format: str = "jpg",
    jpeg_quality: int = 95,
) -> list[Path]:
    _, ImageOps = require_pillow()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extension = image_format.lower().lstrip(".")
    if extension == "jpeg":
        extension = "jpg"

    if extension not in {"jpg", "png", "webp"}:
        raise ValueError("image_format must be one of: jpg, png, webp")

    pil_format = "JPEG" if extension == "jpg" else extension.upper()
    padding = max(2, len(str(len(images))))
    saved_paths: list[Path] = []

    for page_index, image in enumerate(images, start=1):
        page_image = ImageOps.exif_transpose(image)
        if pil_format == "JPEG":
            page_image = page_image.convert("RGB")

        image_path = output_path / (
            f"{stem}_page_{page_index:0{padding}d}.{extension}"
        )

        if pil_format == "JPEG":
            page_image.save(
                image_path,
                format=pil_format,
                quality=jpeg_quality,
                optimize=True,
            )
        else:
            page_image.save(image_path, format=pil_format)

        saved_paths.append(image_path)

    return saved_paths


def pdf_to_image_files(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    dpi: int = 200,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
) -> list[Path]:
    path = Path(pdf_path)
    images = pdf_to_images(path, dpi=dpi)
    image_dir = Path(output_dir) if output_dir else default_pdf_image_dir(path)

    return save_images_to_files(
        images=images,
        output_dir=image_dir,
        stem=path.stem,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
    )


def load_input_images_with_paths(
    input_path: str | Path,
    pdf_dpi: int = 200,
    save_pdf_images: bool = False,
    pdf_image_dir: str | Path | None = None,
    pdf_image_format: str = "jpg",
    jpeg_quality: int = 95,
) -> tuple[list[Any], str, list[Path | None]]:
    Image, _ = require_pillow()

    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        images = pdf_to_images(path, dpi=pdf_dpi)
        image_paths: list[Path | None] = [None] * len(images)

        if save_pdf_images:
            image_dir = (
                Path(pdf_image_dir)
                if pdf_image_dir
                else default_pdf_image_dir(path)
            )
            image_paths = save_images_to_files(
                images=images,
                output_dir=image_dir,
                stem=path.stem,
                image_format=pdf_image_format,
                jpeg_quality=jpeg_quality,
            )

        return images, "PDF", image_paths

    if suffix in IMAGE_EXTENSIONS:
        return [Image.open(path)], "IMAGE", [path]

    raise ValueError(
        f"Unsupported input '{path}'. Use a PDF or image file."
    )


def load_input_images(
    input_path: str | Path,
    pdf_dpi: int = 200,
) -> tuple[list[Any], str]:
    images, source_type, _ = load_input_images_with_paths(
        input_path=input_path,
        pdf_dpi=pdf_dpi,
    )
    return images, source_type



def extract_json_text(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        character = text[index]

        if escape:
            escape = False
            continue

        if character == "\\":
            escape = True
            continue

        if character == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    return text[start:]


def parse_model_json(response: Any) -> dict[str, Any]:
    json_text = extract_json_text(response)

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as first_error:
        try:
            parsed = ast.literal_eval(json_text)
        except Exception:
            return {
                "parse_error": str(first_error),
                "raw_model_response": response,
            }

    if isinstance(parsed, dict):
        return parsed

    return {
        "parse_error": "Model response is valid JSON but not a JSON object.",
        "raw_model_response": response,
    }


def extract_pdf_text_pages_if_available(input_path: str | Path) -> list[str]:
    if Path(input_path).suffix.lower() != ".pdf":
        return []
    try:
        from backend.app.pipeline import extract_pdf_text_pages
    except Exception:
        return []
    return extract_pdf_text_pages(input_path)


def enrich_with_pdf_text_layer(
    page_data: dict[str, Any],
    pdf_text_pages: list[str],
    page_index: int,
) -> dict[str, Any]:
    try:
        from backend.app.pipeline import (
            clean_text_value,
            normalize_invoice_data,
            overlay_invoice_data,
            parse_invoice_text_layer,
        )
    except Exception:
        return page_data

    page_data = normalize_invoice_data(page_data)
    if page_index - 1 >= len(pdf_text_pages):
        return page_data

    page_text = pdf_text_pages[page_index - 1]
    if not clean_text_value(page_text):
        return page_data

    return overlay_invoice_data(page_data, parse_invoice_text_layer(page_text))


def extract_sufficient_pdf_text_layer_data(
    pdf_text_pages: list[str],
    page_index: int,
) -> dict[str, Any] | None:
    try:
        from backend.app.pipeline import (
            clean_text_value,
            is_text_layer_data_sufficient,
            normalize_invoice_data,
            parse_invoice_text_layer,
        )
    except Exception:
        return None

    if page_index - 1 >= len(pdf_text_pages):
        return None

    page_text = pdf_text_pages[page_index - 1]
    if not clean_text_value(page_text):
        return None

    page_data = parse_invoice_text_layer(page_text)
    return normalize_invoice_data(page_data) if is_text_layer_data_sufficient(page_data) else None


def response_to_text(response: Any) -> str:
    if isinstance(response, dict):
        choices = response.get("choices")
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content

    return str(response)


def normalize_server_url(server_url: str) -> str:
    server_url = server_url.rstrip("/")

    if server_url.endswith("/v1"):
        return server_url

    return f"{server_url}/v1"


def get_server_origin(server_url: str) -> str:
    server_url = server_url.rstrip("/")

    if server_url.endswith("/v1"):
        return server_url[:-3].rstrip("/")

    return server_url


def get_chat_completion_urls(server_url: str) -> list[str]:
    origin = get_server_origin(server_url)
    base_url = normalize_server_url(server_url)

    urls = [
        f"{base_url}/chat/completions",
        f"{origin}/chat/completions",
    ]

    return list(dict.fromkeys(urls))


def probe_llama_server(
    server_url: str = "http://127.0.0.1:8080/v1",
    timeout: int = 10,
) -> list[dict[str, Any]]:
    requests = require_requests()
    origin = get_server_origin(server_url)
    base_url = normalize_server_url(server_url)

    urls = [
        origin,
        f"{base_url}/models",
        f"{origin}/v1/models",
        f"{origin}/props",
        f"{origin}/health",
    ]

    results: list[dict[str, Any]] = []

    for url in list(dict.fromkeys(urls)):
        try:
            response = requests.get(url, timeout=timeout)
            content_type = response.headers.get("content-type", "")
            body = response.text[:500]
            if "application/json" in content_type:
                try:
                    body = json.dumps(
                        response.json(),
                        ensure_ascii=False,
                    )[:500]
                except ValueError:
                    pass

            results.append(
                {
                    "url": url,
                    "status_code": response.status_code,
                    "content_type": content_type,
                    "body": body,
                }
            )
        except requests.RequestException as error:
            results.append(
                {
                    "url": url,
                    "error": str(error),
                }
            )

    return results


def run_vintern_gguf(
    llm: Any,
    image: Any,
    prompt: str = INVOICE_PROMPT,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_image_side: int = 1800,
    jpeg_quality: int = 92,
) -> str:
    image_url = image_to_data_uri(
        image=image,
        max_side=max_image_side,
        jpeg_quality=jpeg_quality,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
            ],
        }
    ]

    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as error:
        error_text = str(error)
        if "can only concatenate str" in error_text and "list" in error_text:
            raise RuntimeError(
                "llama-cpp-python is using a text-only chat template, so it "
                "cannot receive an image message. Reload the model with "
                "load_llm(chat_format='mtmd'), then rerun this cell."
            ) from error

        if "Invalid chat handler: mtmd" in error_text:
            raise RuntimeError(
                "Your installed llama-cpp-python does not have the mtmd "
                "multimodal handler. Run this in the same notebook/kernel "
                "environment, restart the kernel, then reload llm:\n"
                f"  {LLAMA_CPP_UPGRADE_COMMAND}"
            ) from error

        raise

    return response_to_text(response)


def run_vintern_server(
    image: Any,
    prompt: str = INVOICE_PROMPT,
    server_url: str = "http://127.0.0.1:8080/v1",
    model: str = "local-model",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_image_side: int = 1800,
    jpeg_quality: int = 92,
    timeout: int = 600,
) -> str:
    requests = require_requests()

    image_url = image_to_data_uri(
        image=image,
        max_side=max_image_side,
        jpeg_quality=jpeg_quality,
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    urls = get_chat_completion_urls(server_url)
    not_found_errors: list[str] = []
    last_response_text = ""

    for url in urls:
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as error:
            raise RuntimeError(
                "Cannot connect to llama.cpp server. Start it in another terminal:\n"
                "  llama-server -hf rootonchair/Vintern-1B-v3_5-GGUF-ext:Q4_K_M\n\n"
                f"Server URL used by the pipeline: {server_url}"
            ) from error

        last_response_text = response.text

        if response.status_code == 404:
            not_found_errors.append(f"{url} -> HTTP 404")
            continue

        if not response.ok:
            raise RuntimeError(
                "llama.cpp server returned an error:\n"
                f"URL: {url}\n"
                f"HTTP {response.status_code}\n{response.text}"
            )

        return response_to_text(response.json())

    raise RuntimeError(
        "The server is reachable, but no OpenAI-compatible chat endpoint was found.\n"
        f"Tried: {not_found_errors}\n\n"
        "Check that the terminal is running `llama-server`, not `llama-cli`, "
        "and that no other app is using port 8080. A recent llama.cpp server "
        "should respond at:\n"
        "  http://127.0.0.1:8080/v1/models\n"
        "  http://127.0.0.1:8080/v1/chat/completions\n\n"
        "In the notebook, run `probe_llama_server(SERVER_URL)` to inspect what "
        "is actually listening on that port.\n\n"
        f"Last response body:\n{last_response_text[:1000]}"
    )


def scan_invoice_file(
    input_path: str | Path,
    llm: Any,
    prompt: str = INVOICE_PROMPT,
    pdf_dpi: int = 200,
    save_pdf_images: bool = False,
    pdf_image_dir: str | Path | None = None,
    pdf_image_format: str = "jpg",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_image_side: int = 1800,
    jpeg_quality: int = 92,
) -> list[dict[str, Any]]:
    images, source_type, image_paths = load_input_images_with_paths(
        input_path=input_path,
        pdf_dpi=pdf_dpi,
        save_pdf_images=save_pdf_images,
        pdf_image_dir=pdf_image_dir,
        pdf_image_format=pdf_image_format,
        jpeg_quality=jpeg_quality,
    )
    results: list[dict[str, Any]] = []
    pdf_text_pages = extract_pdf_text_pages_if_available(input_path)

    for page_index, image in enumerate(images, start=1):
        print(f"Processing page {page_index}/{len(images)}...")
        raw_response = None
        page_data = extract_sufficient_pdf_text_layer_data(pdf_text_pages, page_index)

        if page_data is None:
            print("Text layer chưa đủ dữ liệu, chạy model vision...")
            raw_response = run_vintern_gguf(
                llm=llm,
                image=image,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                max_image_side=max_image_side,
                jpeg_quality=jpeg_quality,
            )
            page_data = parse_model_json(raw_response)
            page_data = enrich_with_pdf_text_layer(page_data, pdf_text_pages, page_index)

        if not isinstance(page_data.get("metadata"), dict):
            page_data["metadata"] = {}

        page_data["metadata"].update(
            {
                "source_type": source_type,
                "page_count": len(images),
                "ocr_processed": True,
                "vision_model_used": raw_response is not None,
            }
        )

        page_image_path = image_paths[page_index - 1]
        if page_image_path:
            page_data["metadata"]["page_image_path"] = str(page_image_path)

        results.append(
            {
                "page": page_index,
                "data": page_data,
                **({"raw_response": raw_response} if raw_response is not None else {}),
            }
        )

    return results


def scan_invoice_file_server(
    input_path: str | Path,
    prompt: str = INVOICE_PROMPT,
    server_url: str = "http://127.0.0.1:8080/v1",
    model: str = "local-model",
    pdf_dpi: int = 200,
    save_pdf_images: bool = False,
    pdf_image_dir: str | Path | None = None,
    pdf_image_format: str = "jpg",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_image_side: int = 1800,
    jpeg_quality: int = 92,
    timeout: int = 600,
) -> list[dict[str, Any]]:
    images, source_type, image_paths = load_input_images_with_paths(
        input_path=input_path,
        pdf_dpi=pdf_dpi,
        save_pdf_images=save_pdf_images,
        pdf_image_dir=pdf_image_dir,
        pdf_image_format=pdf_image_format,
        jpeg_quality=jpeg_quality,
    )
    results: list[dict[str, Any]] = []
    pdf_text_pages = extract_pdf_text_pages_if_available(input_path)

    for page_index, image in enumerate(images, start=1):
        print(f"Processing page {page_index}/{len(images)}...")
        raw_response = None
        page_data = extract_sufficient_pdf_text_layer_data(pdf_text_pages, page_index)

        if page_data is None:
            print("Text layer chưa đủ dữ liệu, chạy llama.cpp server...")
            raw_response = run_vintern_server(
                image=image,
                prompt=prompt,
                server_url=server_url,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                max_image_side=max_image_side,
                jpeg_quality=jpeg_quality,
                timeout=timeout,
            )
            page_data = parse_model_json(raw_response)
            page_data = enrich_with_pdf_text_layer(page_data, pdf_text_pages, page_index)

        if not isinstance(page_data.get("metadata"), dict):
            page_data["metadata"] = {}

        page_data["metadata"].update(
            {
                "source_type": source_type,
                "page_count": len(images),
                "ocr_processed": True,
                "vision_model_used": raw_response is not None,
            }
        )

        page_image_path = image_paths[page_index - 1]
        if page_image_path:
            page_data["metadata"]["page_image_path"] = str(page_image_path)

        results.append(
            {
                "page": page_index,
                "data": page_data,
                **({"raw_response": raw_response} if raw_response is not None else {}),
            }
        )

    return results


def save_results(
    results: list[dict[str, Any]],
    output_path: str | Path,
    include_raw_response: bool = False,
) -> None:
    output = []

    for result in results:
        row = dict(result)
        if not include_raw_response:
            row.pop("raw_response", None)
        output.append(row)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local CPU invoice OCR with Vintern GGUF."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an invoice image or PDF.",
    )
    parser.add_argument(
        "--backend",
        default="server",
        choices=["server", "python"],
        help="Inference backend. Use 'server' for llama.cpp server.",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path. Defaults to <input>.json.",
    )
    parser.add_argument(
        "--repo-id",
        default=MODEL_REPO_ID,
        help="Hugging Face model repository.",
    )
    parser.add_argument(
        "--quant",
        default="Q4_K_M",
        choices=sorted(QUANT_FILES),
        help="Quantization to download/use.",
    )
    parser.add_argument(
        "--filename",
        help="Exact GGUF filename. Overrides --quant.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="CPU threads for llama.cpp. Defaults to CPU count minus one.",
    )
    parser.add_argument(
        "--chat-format",
        default="mtmd",
        help="llama-cpp-python chat format. Keep 'mtmd' for multimodal GGUF.",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8080/v1",
        help="OpenAI-compatible llama.cpp server URL.",
    )
    parser.add_argument(
        "--server-model",
        default="local-model",
        help="Model name sent to the OpenAI-compatible server.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="HTTP timeout in seconds for server backend.",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=8192,
        help="Context length.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum generated tokens per page.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Keep 0 for deterministic extraction.",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=200,
        help="DPI used when rendering PDF pages.",
    )
    parser.add_argument(
        "--save-pdf-images",
        action="store_true",
        help="Save rendered PDF pages as images before OCR.",
    )
    parser.add_argument(
        "--pdf-image-dir",
        help="Directory for rendered PDF page images. Defaults to <pdf>_images.",
    )
    parser.add_argument(
        "--pdf-image-format",
        default="jpg",
        choices=["jpg", "png", "webp"],
        help="Image format used when saving rendered PDF pages.",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1800,
        help="Resize page/image so the longest side is at most this value.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality for the image sent to llama.cpp.",
    )
    parser.add_argument(
        "--include-raw-response",
        action="store_true",
        help="Include raw model text in the output JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show llama.cpp logs.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".json")

    if args.backend == "python":
        llm = load_llm(
            repo_id=args.repo_id,
            quant=args.quant,
            filename=args.filename,
            n_ctx=args.n_ctx,
            n_threads=args.threads,
            chat_format=args.chat_format,
            verbose=args.verbose,
        )

        results = scan_invoice_file(
            input_path=input_path,
            llm=llm,
            pdf_dpi=args.pdf_dpi,
            save_pdf_images=args.save_pdf_images,
            pdf_image_dir=args.pdf_image_dir,
            pdf_image_format=args.pdf_image_format,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_image_side=args.max_image_side,
            jpeg_quality=args.jpeg_quality,
        )
    else:
        results = scan_invoice_file_server(
            input_path=input_path,
            server_url=args.server_url,
            model=args.server_model,
            pdf_dpi=args.pdf_dpi,
            save_pdf_images=args.save_pdf_images,
            pdf_image_dir=args.pdf_image_dir,
            pdf_image_format=args.pdf_image_format,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_image_side=args.max_image_side,
            jpeg_quality=args.jpeg_quality,
            timeout=args.timeout,
        )

    save_results(
        results=results,
        output_path=output_path,
        include_raw_response=args.include_raw_response,
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
