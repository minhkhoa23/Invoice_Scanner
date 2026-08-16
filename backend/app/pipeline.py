from __future__ import annotations

import ast
import base64
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:  # PyMuPDF exposes both import styles across versions.
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_REPO_ID = "rootonchair/Vintern-1B-v3_5-GGUF-ext"
MODEL_QUANT = "Q4_K_M"
MMPROJ_FILENAME = "mmproj-Vintern-1B-v3_5-Q8_0.gguf"
MMPROJ_PATH = PROJECT_ROOT / "models" / MMPROJ_FILENAME

SYSTEM_MESSAGE = (
    "Bạn là một mô hình trí tuệ nhân tạo đa phương thức Tiếng Việt có tên gọi "
    "là Vintern, được phát triển bởi người Việt. Bạn là một trợ lý trí tuệ "
    "nhân tạo hữu ích và không gây hại."
)
MTMD_MEDIA_MARKER = "<__media__>"
VINTERN_IMAGE_MARKER = "<image>"

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
        "pdf_text_layer_used": "boolean | null",
    },
}

INVOICE_PROMPT = f"""
Bạn là hệ thống OCR và trích xuất thông tin từ hóa đơn Việt Nam.

Trả về duy nhất một JSON object hợp lệ, không dùng markdown, không giải thích.

Quy tắc bắt buộc:
- JSON root phải có đúng các khóa cấp 1: invoice, seller, buyer, shipping, items, totals, signature, metadata.
- Không trả JSON phẳng kiểu seller_name, buyer_tax_code, invoice_date ở cấp root.
- Mọi thông tin người bán phải nằm trong seller; người mua trong buyer; tổng tiền trong totals.
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


class OCRPipelineError(RuntimeError):
    """Raised when local OCR processing cannot complete."""


@dataclass(frozen=True)
class OCRConfig:
    server_url: str = "http://127.0.0.1:8081/v1"
    model: str = "local-model"
    pdf_dpi: int = 300
    max_image_side: int = 2600
    jpeg_quality: int = 95
    max_tokens: int = 2048
    temperature: float = 0.0
    request_timeout: int = 600
    use_image_enhancement: bool = True
    use_pdf_text_layer: bool = True
    pdf_text_strategy: str = "assist"
    include_raw_response: bool = False


PDF_TEXT_STRATEGIES = {"assist", "fast", "off"}


def normalize_pdf_text_strategy(value: str | None) -> str:
    strategy = (value or "assist").strip().lower()
    return strategy if strategy in PDF_TEXT_STRATEGIES else "assist"


def ocr_config_from_env() -> OCRConfig:
    return OCRConfig(
        server_url=os.getenv("OCR_SERVER_URL", "http://127.0.0.1:8081/v1"),
        model=os.getenv("OCR_SERVER_MODEL", "local-model"),
        pdf_dpi=int(os.getenv("OCR_PDF_DPI", "300")),
        max_image_side=int(os.getenv("OCR_MAX_IMAGE_SIDE", "2600")),
        jpeg_quality=int(os.getenv("OCR_JPEG_QUALITY", "95")),
        max_tokens=int(os.getenv("OCR_MAX_TOKENS", "2048")),
        temperature=float(os.getenv("OCR_TEMPERATURE", "0.0")),
        request_timeout=int(os.getenv("OCR_REQUEST_TIMEOUT", "600")),
        use_image_enhancement=os.getenv("OCR_USE_IMAGE_ENHANCEMENT", "true").lower()
        != "false",
        use_pdf_text_layer=os.getenv("OCR_USE_PDF_TEXT_LAYER", "true").lower()
        != "false",
        pdf_text_strategy=normalize_pdf_text_strategy(
            os.getenv("OCR_PDF_TEXT_STRATEGY", "assist")
        ),
        include_raw_response=os.getenv("OCR_INCLUDE_RAW_RESPONSE", "false").lower()
        == "true",
    )


def server_origin(server_url: str) -> str:
    server_url = server_url.rstrip("/")
    return server_url[:-3].rstrip("/") if server_url.endswith("/v1") else server_url


def normalized_v1_url(server_url: str) -> str:
    server_url = server_url.rstrip("/")
    return server_url if server_url.endswith("/v1") else f"{server_url}/v1"


def completion_url(server_url: str) -> str:
    return f"{server_origin(server_url)}/completion"


def chat_completion_urls(server_url: str) -> list[str]:
    origin = server_origin(server_url)
    base = normalized_v1_url(server_url)
    return list(dict.fromkeys([f"{base}/chat/completions", f"{origin}/chat/completions"]))


def probe_server(server_url: str, timeout: int = 5) -> dict[str, Any]:
    origin = server_origin(server_url)
    urls = [
        origin,
        f"{origin}/v1/models",
        f"{origin}/props",
        f"{origin}/health",
    ]
    checks: list[dict[str, Any]] = []

    for url in urls:
        try:
            response = requests.get(url, timeout=timeout)
            body = response.text[:500]
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    body = json.dumps(response.json(), ensure_ascii=False)[:500]
                except ValueError:
                    pass
            checks.append(
                {
                    "url": url,
                    "ok": response.ok,
                    "status_code": response.status_code,
                    "content_type": content_type,
                    "body": body,
                }
            )
        except requests.RequestException as error:
            checks.append({"url": url, "ok": False, "error": str(error)})

    return {
        "server_url": server_url,
        "origin": origin,
        "reachable": any(item.get("ok") for item in checks),
        "checks": checks,
        "server_command": build_llama_server_command(),
    }


def build_llama_server_command() -> str:
    return (
        f'llama-server -hf {MODEL_REPO_ID}:{MODEL_QUANT} '
        f'--mmproj "{MMPROJ_PATH}" '
        "--chat-template vicuna "
        "--port 8081"
    )


def pdf_to_images(pdf_path: str | Path, dpi: int = 200) -> list[Image.Image]:
    document = fitz.open(pdf_path)
    images: list[Image.Image] = []
    matrix = fitz.Matrix(dpi / 72, dpi / 72)

    for page_index in range(len(document)):
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        images.append(image)

    document.close()
    return images


def save_images_to_files(
    images: list[Image.Image],
    output_dir: str | Path,
    stem: str,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extension = image_format.lower().lstrip(".")
    pil_format = "JPEG" if extension in {"jpg", "jpeg"} else extension.upper()
    extension = "jpg" if extension == "jpeg" else extension
    padding = max(2, len(str(len(images))))
    paths: list[Path] = []

    for page_index, image in enumerate(images, start=1):
        page_image = ImageOps.exif_transpose(image)
        if pil_format == "JPEG":
            page_image = page_image.convert("RGB")

        image_path = output_path / f"{stem}_page_{page_index:0{padding}d}.{extension}"
        if pil_format == "JPEG":
            page_image.save(
                image_path,
                format=pil_format,
                quality=jpeg_quality,
                optimize=True,
            )
        else:
            page_image.save(image_path, format=pil_format)
        paths.append(image_path)

    return paths


def load_input_images(
    input_path: str | Path,
    pdf_dpi: int = 200,
    save_pdf_images: bool = True,
    pdf_image_dir: str | Path | None = None,
) -> tuple[list[Image.Image], str, list[Path | None]]:
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix == ".pdf":
        images = pdf_to_images(input_path, dpi=pdf_dpi)
        image_dir = (
            Path(pdf_image_dir)
            if pdf_image_dir
            else input_path.with_name(f"{input_path.stem}_images")
        )
        image_paths: list[Path | None] = (
            save_images_to_files(images, image_dir, input_path.stem)
            if save_pdf_images
            else [None] * len(images)
        )
        return images, "PDF", image_paths

    if suffix in IMAGE_EXTENSIONS:
        image = ImageOps.exif_transpose(Image.open(input_path)).convert("RGB")
        return [image], "IMAGE", [input_path]

    raise OCRPipelineError(f"File không hỗ trợ: {input_path}. Hãy dùng PDF hoặc ảnh.")


def enhance_invoice_image(image: Image.Image) -> Image.Image:
    image = ImageOps.autocontrast(image, cutoff=1)
    image = ImageEnhance.Contrast(image).enhance(1.15)
    image = ImageEnhance.Sharpness(image).enhance(1.25)
    return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))


def is_probably_blank_image(
    image: Image.Image,
    white_threshold: int = 245,
    ink_ratio_threshold: float = 0.001,
) -> bool:
    grayscale = ImageOps.exif_transpose(image).convert("L")
    grayscale.thumbnail((400, 400), Image.Resampling.LANCZOS)
    pixels = list(grayscale.getdata())
    if not pixels:
        return True
    ink_pixels = sum(1 for pixel in pixels if pixel < white_threshold)
    return (ink_pixels / len(pixels)) < ink_ratio_threshold


def normalize_image(
    image: Image.Image,
    max_side: int = 1800,
    use_image_enhancement: bool = True,
) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if use_image_enhancement:
        image = enhance_invoice_image(image)
    return image


def image_to_base64_jpeg(
    image: Image.Image,
    max_side: int = 1800,
    jpeg_quality: int = 92,
    use_image_enhancement: bool = True,
) -> str:
    image = normalize_image(
        image,
        max_side=max_side,
        use_image_enhancement=use_image_enhancement,
    )
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def image_to_data_uri(
    image: Image.Image,
    max_side: int,
    jpeg_quality: int,
    use_image_enhancement: bool,
) -> str:
    encoded = image_to_base64_jpeg(
        image=image,
        max_side=max_side,
        jpeg_quality=jpeg_quality,
        use_image_enhancement=use_image_enhancement,
    )
    return f"data:image/jpeg;base64,{encoded}"


def get_server_props(server_url: str, timeout: int = 10) -> dict[str, Any]:
    response = requests.get(f"{server_origin(server_url)}/props", timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_server_media_marker(server_url: str, timeout: int = 10) -> str:
    try:
        props = get_server_props(server_url, timeout=timeout)
    except requests.RequestException:
        return MTMD_MEDIA_MARKER

    marker = props.get("media_marker")
    return marker if isinstance(marker, str) and marker else MTMD_MEDIA_MARKER


def strip_media_markers(prompt: str | None) -> str:
    prompt = prompt or INVOICE_PROMPT
    prompt = re.sub(r"<__media[^>]*__>\s*", "", prompt)
    return prompt.replace(VINTERN_IMAGE_MARKER, "").strip()


def build_completion_prompt(prompt: str | None, server_url: str) -> str:
    media_marker = get_server_media_marker(server_url)
    clean_prompt = strip_media_markers(prompt)
    return (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>"
        f"<|im_start|>user\n{media_marker}\n{VINTERN_IMAGE_MARKER}\n"
        f"{clean_prompt}<|im_end|>"
        "<|im_start|>assistant\n"
    )


def response_to_text(response_json: Any) -> str:
    if isinstance(response_json, dict):
        content = response_json.get("content")
        if isinstance(content, str):
            return content

        completion = response_json.get("completion")
        if isinstance(completion, str):
            return completion

        choices = response_json.get("choices")
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content

    return str(response_json)


def run_completion_endpoint(
    image: Image.Image,
    config: OCRConfig,
    prompt: str | None = None,
) -> str:
    prompt_text = build_completion_prompt(prompt, server_url=config.server_url)
    image_base64 = image_to_base64_jpeg(
        image,
        max_side=config.max_image_side,
        jpeg_quality=config.jpeg_quality,
        use_image_enhancement=config.use_image_enhancement,
    )
    payload = {
        "prompt": {
            "prompt_string": prompt_text,
            "multimodal_data": [image_base64],
        },
        "n_predict": config.max_tokens,
        "temperature": config.temperature,
        "stream": False,
        "cache_prompt": False,
        "ignore_eos": True,
        "stop": ["<|im_end|>", "<|endoftext|>", "</s>", "\n\n*"],
        "repeat_penalty": 1.05,
    }

    response = requests.post(
        completion_url(config.server_url),
        json=payload,
        timeout=config.request_timeout,
    )
    if response.status_code == 404:
        raise OCRPipelineError("Completion endpoint not found.")
    if not response.ok:
        error_text = response.text[:1200]
        if "mmproj" in error_text.lower() or "image input is not supported" in error_text.lower():
            raise OCRPipelineError(
                "Server chưa load multimodal projector (mmproj), nên chưa đọc được ảnh. "
                "Hãy dừng llama-server cũ và chạy lại command có --mmproj."
            )
        raise OCRPipelineError(
            f"llama.cpp server lỗi tại {completion_url(config.server_url)}: "
            f"HTTP {response.status_code}\n{error_text}"
        )

    output_text = response_to_text(response.json())
    if not output_text.strip():
        raise OCRPipelineError("Server trả về nội dung rỗng.")
    return output_text


def run_chat_endpoint(
    image: Image.Image,
    config: OCRConfig,
    prompt: str | None = None,
) -> str:
    image_url = image_to_data_uri(
        image=image,
        max_side=config.max_image_side,
        jpeg_quality=config.jpeg_quality,
        use_image_enhancement=config.use_image_enhancement,
    )
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or INVOICE_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }

    last_error = ""
    for url in chat_completion_urls(config.server_url):
        response = requests.post(url, json=payload, timeout=config.request_timeout)
        if response.status_code == 404:
            last_error = f"{url} -> HTTP 404"
            continue
        if not response.ok:
            raise OCRPipelineError(
                f"llama.cpp server lỗi tại {url}: HTTP {response.status_code}\n"
                f"{response.text[:1200]}"
            )
        output_text = response_to_text(response.json())
        if output_text.strip():
            return output_text

    raise OCRPipelineError(
        "Không tìm thấy endpoint chat/completions tương thích. "
        f"Chi tiết cuối: {last_error}"
    )


def run_vintern_server(
    image: Image.Image,
    config: OCRConfig,
    prompt: str | None = None,
) -> str:
    try:
        return run_completion_endpoint(image=image, config=config, prompt=prompt)
    except requests.RequestException as error:
        raise OCRPipelineError(
            "Không kết nối được llama.cpp server. Hãy chạy terminal riêng:\n"
            f"{build_llama_server_command()}"
        ) from error
    except OCRPipelineError as completion_error:
        if "Completion endpoint not found" not in str(completion_error):
            raise

    try:
        return run_chat_endpoint(image=image, config=config, prompt=prompt)
    except requests.RequestException as error:
        raise OCRPipelineError(
            "Không kết nối được llama.cpp server. Hãy chạy terminal riêng:\n"
            f"{build_llama_server_command()}"
        ) from error


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
                return text[start : index + 1]

    return text[start:]


def parse_model_json(response: Any) -> dict[str, Any]:
    json_text = extract_json_text(response)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as first_error:
        try:
            parsed = ast.literal_eval(json_text)
        except Exception:
            return {"parse_error": str(first_error), "raw_model_response": response}

    if isinstance(parsed, dict):
        return parsed

    return {
        "parse_error": "Model response is valid JSON but not an object.",
        "raw_model_response": response,
    }


def blank_invoice_json() -> dict[str, Any]:
    return {
        "invoice": {
            "invoice_type": None,
            "invoice_number": None,
            "series": None,
            "invoice_date": None,
            "tax_authority_code": None,
            "currency": None,
            "payment_method": None,
        },
        "seller": {
            "name": None,
            "english_name": None,
            "tax_code": None,
            "address": None,
            "phone": [],
            "fax": [],
            "email": None,
            "website": None,
        },
        "buyer": {
            "name": None,
            "company_name": None,
            "tax_code": None,
            "address": None,
            "id_card": None,
            "passport_number": None,
            "account_number": None,
            "budgetary_unit_code": None,
        },
        "shipping": {"ship_from_warehouse": None},
        "items": [],
        "totals": {
            "subtotal": None,
            "vat_rate": None,
            "vat_amount": None,
            "total_payment": None,
            "amount_in_words": None,
        },
        "signature": {
            "is_valid": None,
            "signed_by": None,
            "signed_date": None,
        },
        "metadata": {
            "source_type": None,
            "page_count": None,
            "ocr_processed": True,
            "pdf_text_layer_used": False,
        },
    }


FLAT_FIELD_MAP = {
    "type": ("invoice", "invoice_type"),
    "invoice_type": ("invoice", "invoice_type"),
    "number": ("invoice", "invoice_number"),
    "invoice_number": ("invoice", "invoice_number"),
    "series": ("invoice", "series"),
    "date": ("invoice", "invoice_date"),
    "invoice_date": ("invoice", "invoice_date"),
    "tax_authority_code": ("invoice", "tax_authority_code"),
    "currency": ("invoice", "currency"),
    "payment_method": ("invoice", "payment_method"),
    "tax_code": ("seller", "tax_code"),
    "address": ("seller", "address"),
    "phone": ("seller", "phone"),
    "fax": ("seller", "fax"),
    "email": ("seller", "email"),
    "website": ("seller", "website"),
    "seller_name": ("seller", "name"),
    "seller_english_name": ("seller", "english_name"),
    "seller_tax_code": ("seller", "tax_code"),
    "seller_address": ("seller", "address"),
    "seller_phone": ("seller", "phone"),
    "seller_fax": ("seller", "fax"),
    "seller_email": ("seller", "email"),
    "seller_website": ("seller", "website"),
    "buyer_name": ("buyer", "name"),
    "buyer_company_name": ("buyer", "company_name"),
    "buyer_tax_code": ("buyer", "tax_code"),
    "buyer_address": ("buyer", "address"),
    "buyer_id_card": ("buyer", "id_card"),
    "buyer_passport_number": ("buyer", "passport_number"),
    "buyer Passport_number": ("buyer", "passport_number"),
    "buyer_account_number": ("buyer", "account_number"),
    "buyer_budget_unit_code": ("buyer", "budgetary_unit_code"),
    "ship_from_warehouse": ("shipping", "ship_from_warehouse"),
    "subtotal": ("totals", "subtotal"),
    "vat_rate": ("totals", "vat_rate"),
    "vat_amount": ("totals", "vat_amount"),
    "total_payment": ("totals", "total_payment"),
    "amount_in_words": ("totals", "amount_in_words"),
    "signature_is_valid": ("signature", "is_valid"),
    "signed_by": ("signature", "signed_by"),
    "signed_date": ("signature", "signed_date"),
}

SECTION_FIELD_ALIASES = {
    "invoice": {
        "type": "invoice_type",
        "number": "invoice_number",
        "date": "invoice_date",
    }
}

ITEM_FIELD_ALIASES = {
    "name": "description",
    "service_name": "description",
    "goods_name": "description",
    "product_name": "description",
    "line_description": "description",
    "tax_rate": "vat_rate",
    "tax_percent": "vat_rate",
    "tax_amount": "vat_amount",
    "line_tax": "vat_amount",
    "before_tax_amount": "taxable_amount",
    "amount_before_tax": "taxable_amount",
    "line_amount_before_tax": "taxable_amount",
    "line_total": "total_amount",
    "total": "total_amount",
    "payment_amount": "total_amount",
    "container_no": "container_number",
    "container_code": "container_number",
}

ITEM_NUMBER_FIELDS = {
    "line_number",
    "quantity",
    "unit_price",
    "amount",
    "taxable_amount",
    "vat_rate",
    "vat_amount",
    "total_amount",
}

NUMERIC_FIELD_NAMES = ITEM_NUMBER_FIELDS | {
    "subtotal",
    "vat_rate",
    "vat_amount",
    "total_payment",
    "page_count",
}

PLACEHOLDER_TEXT_VALUES = {
    "-",
    ".",
    "\\",
    "n/a",
    "na",
    "none",
    "null",
    "string",
    "number",
    "boolean",
}


def is_placeholder_text(value: str) -> bool:
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    return (
        normalized in PLACEHOLDER_TEXT_VALUES
        or ("|" in normalized and "null" in normalized)
    )


def normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    normalized_values = []
    for item in values:
        if isinstance(item, str):
            item = clean_text_value(item)
        if item is not None and item != "":
            normalized_values.append(item)
    return normalized_values


def normalize_date_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})\s*", value)
    if not match:
        return value
    day, month, year = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def apply_section_aliases(section: str, value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    value = dict(value)
    for alias, canonical in SECTION_FIELD_ALIASES.get(section, {}).items():
        if alias in value and canonical not in value:
            value[canonical] = value[alias]
    return value


def merge_known_fields(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, str):
            value = clean_text_value(value)
        if key not in target or value is None:
            continue
        if key.endswith("date"):
            value = normalize_date_value(value)
        if isinstance(target[key], dict) and isinstance(value, dict):
            merge_known_fields(target[key], value)
        elif key in {"phone", "fax"}:
            target[key] = normalize_list(value)
        elif key == "items" and isinstance(value, list):
            target[key] = [normalize_item(item) for item in value if isinstance(item, dict)]
        elif key in NUMERIC_FIELD_NAMES:
            target[key] = parse_vn_percent(value) if key == "vat_rate" else parse_vn_number(value)
        else:
            target[key] = value


def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    item = dict(item)
    for alias, canonical in ITEM_FIELD_ALIASES.items():
        if alias in item and canonical not in item:
            item[canonical] = item[alias]

    if item.get("taxable_amount") is None and item.get("amount") is not None:
        item["taxable_amount"] = item["amount"]
    if item.get("amount") is None and item.get("taxable_amount") is not None:
        item["amount"] = item["taxable_amount"]

    template = {
        "line_number": None,
        "description": None,
        "description_lines": [],
        "item_type": None,
        "container_number": None,
        "unit": None,
        "quantity": None,
        "unit_price": None,
        "amount": None,
        "taxable_amount": None,
        "vat_rate": None,
        "vat_amount": None,
        "total_amount": None,
    }
    merge_known_fields(template, item)

    description = clean_text_value(template["description"])
    description_lines = normalize_list(template["description_lines"])
    description_lines = [
        clean_text_value(line) for line in description_lines if clean_text_value(line)
    ]
    if not description and description_lines:
        description = clean_text_value(" ".join(str(line) for line in description_lines))
    if description and not description_lines:
        description_lines = [description]
    template["description"] = description
    template["description_lines"] = description_lines

    if not template["container_number"]:
        template["container_number"] = extract_container_number(description)
    if not template["item_type"]:
        template["item_type"] = infer_item_type(description)

    for field in ITEM_NUMBER_FIELDS:
        if field == "vat_rate":
            template[field] = parse_vn_percent(template[field])
        else:
            template[field] = parse_vn_number(template[field])

    if template["taxable_amount"] is None and template["amount"] is not None:
        template["taxable_amount"] = template["amount"]
    if template["amount"] is None and template["taxable_amount"] is not None:
        template["amount"] = template["taxable_amount"]

    return template


def normalize_invoice_data(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or "parse_error" in data:
        return data

    normalized = blank_invoice_json()

    for section in [
        "invoice",
        "seller",
        "buyer",
        "shipping",
        "totals",
        "signature",
        "metadata",
    ]:
        value = data.get(section)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            value = value[0]
        value = apply_section_aliases(section, value)
        if isinstance(value, dict):
            merge_known_fields(normalized[section], value)

    if isinstance(data.get("items"), list):
        normalized["items"] = [
            normalize_item(item) for item in data["items"] if isinstance(item, dict)
        ]

    for flat_key, (section, field) in FLAT_FIELD_MAP.items():
        if flat_key in data and data[flat_key] is not None:
            value = data[flat_key]
            if field.endswith("date"):
                value = normalize_date_value(value)
            if field in {"phone", "fax"}:
                value = normalize_list(value)
            normalized[section][field] = value

    return normalized


def clean_text_value(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).replace("\xa0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s*\n\s*", " ", value)
    value = value.strip(" :-")
    if is_placeholder_text(value):
        return None
    return value or None


def regex_value(
    pattern: str,
    text: str,
    flags: int = re.IGNORECASE | re.DOTALL,
    group: int = 1,
) -> str | None:
    match = re.search(pattern, text, flags)
    if not match:
        return None
    return clean_text_value(match.group(group))


def normalize_tax_code(value: Any) -> str | None:
    value = clean_text_value(value)
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    return digits or value


def parse_vn_number(value: Any) -> Any:
    if value is None or isinstance(value, (int, float)):
        return value

    text = re.sub(r"[^0-9,\.\-]", "", str(value).strip())
    if not text:
        return None
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    elif text.count(".") > 1:
        text = text.replace(".", "")
    elif "." in text:
        left, right = text.split(".", 1)
        if len(right) == 3:
            text = left + right

    try:
        number = float(text)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def parse_vn_percent(value: Any) -> Any:
    value = clean_text_value(value)
    if not value:
        return None
    return parse_vn_number(value.replace("%", ""))


def extract_container_number(value: Any) -> str | None:
    value = clean_text_value(value)
    if not value:
        return None
    match = re.search(r"\b[A-Z]{4}\d{7}\b", value.upper())
    return match.group(0) if match else None


def infer_item_type(description: Any) -> str | None:
    description = clean_text_value(description)
    if not description:
        return None
    lowered = description.lower()
    if any(keyword in lowered for keyword in ["phụ phí", "phí", "fee", "surcharge"]):
        return "FEE"
    if any(keyword in lowered for keyword in ["dịch vụ", "service"]):
        return "SERVICE"
    return "GOODS"


def normalized_text_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        normalized = clean_text_value(line.replace("\xa0", " "))
        if normalized:
            lines.append(normalized)
    return lines


def strip_accents(value: Any) -> str:
    value = str(value).replace("đ", "d").replace("Đ", "D")
    return "".join(
        character
        for character in unicodedata.normalize("NFD", value)
        if unicodedata.category(character) != "Mn"
    )


def compact_match_text(value: Any) -> str:
    value = clean_text_value(value) or ""
    value = strip_accents(value).lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def plain_layout_text(value: Any) -> str:
    value = clean_text_value(value) or ""
    value = re.sub(r"\s*\|\s*", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def split_layout_cells(value: Any) -> list[str]:
    value = clean_text_value(value)
    if not value:
        return []
    return [
        cell
        for cell in (clean_text_value(part) for part in re.split(r"\s*\|\s*", value))
        if cell
    ]


KNOWN_LABEL_COMPACT_MARKERS = {
    "masothue",
    "taxcode",
    "vatcode",
    "diachi",
    "address",
    "sotaikhoan",
    "bankac",
    "accountno",
    "hinhthucthanhtoan",
    "paymentmethod",
    "modeofpayment",
    "tendonvi",
    "company",
    "companysname",
    "hovatennguoimuahang",
    "hotennguoimuahang",
    "nguoimuahang",
    "customer",
    "agent",
    "donvibanhang",
    "seller",
    "dienthoai",
    "tongdai",
    "callcenter",
    "tel",
    "email",
    "website",
    "sodinhdanh",
    "sohochieu",
    "maquanhe",
    "tygia",
    "exchangerate",
    "donvitiente",
    "currency",
    "tentau",
    "nameofvessel",
    "kho",
    "shipfromwarehouse",
    "kyhieu",
    "serial",
    "macoquanthue",
    "macqt",
    "mcqt",
    "mccqt",
    "ngay",
    "date",
}


def is_known_label_cell(value: Any) -> bool:
    compact = compact_match_text(value)
    if not compact:
        return False
    return any(marker in compact for marker in KNOWN_LABEL_COMPACT_MARKERS)


def is_provider_noise_line(value: Any) -> bool:
    compact = compact_match_text(value)
    return bool(
        "donvicungcap" in compact
        or ("giaiphap" in compact and "hoadondientu" in compact)
        or ("duoccungcap" in compact and "mst" in compact)
    )


def is_table_header_line(value: Any) -> bool:
    compact = compact_match_text(value)
    return bool(
        ("stt" in compact or "no" in compact)
        and (
            "tenhang" in compact
            or "description" in compact
            or "hanghoa" in compact
            or "dichvu" in compact
        )
    )


def is_table_header_guide_line(value: Any) -> bool:
    compact = compact_match_text(value)
    cells = split_layout_cells(value)
    if not compact:
        return False
    if is_table_total_boundary(value):
        return False
    if is_amount_column_marker(value):
        return True
    if re.fullmatch(r"(?:\(?\d+\)?|x|=)+", compact):
        return True
    if cells and all(
        re.fullmatch(r"\(?\d+\)?|x|=", compact_match_text(cell)) for cell in cells
    ):
        return True
    return bool(
        any(
            marker in compact
            for marker in [
                "description",
                "quantity",
                "unitprice",
                "amount",
                "vatrate",
                "vatamount",
            ]
        )
        and not has_numeric_token(value)
    )


def line_matches_compact(value: Any, *patterns: str) -> bool:
    compact = compact_match_text(value)
    return any(re.search(pattern, compact) for pattern in patterns)


def clean_labeled_value(value: Any) -> str | None:
    value = clean_text_value(value)
    if not value:
        return None
    value = re.sub(
        r"^\s*\)?\s*[:：]?\s*(?:\([^)]*\)\s*)?\)?\s*[:：]?\s*",
        "",
        value,
    ).strip()
    value = re.split(
        r"\s+(?:Mã\s*số\s*thuế|MST|Tax\s*code|VAT\s*code|Địa\s*chỉ|Address|"
        r"Số\s*tài\s*khoản|Bank\s*A/C|A/C\s*No\.?|Hình\s*thức\s*thanh\s*toán|"
        r"Payment\s*method|Tên\s*đơn\s*vị|Company|Điện\s*thoại|Tel|Email|Website|Tại)\s*"
        r"(?:\([^)]*\))?\s*:",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = re.sub(r"(?:\s+[a-z]){1,4}$", "", value)
    value = value.strip(" ,;|")
    return clean_text_value(value)


def value_after_label_in_cell(cell: str, label_pattern: str) -> str | None:
    match = re.search(label_pattern, cell, re.IGNORECASE)
    if not match:
        return None
    return clean_labeled_value(cell[match.end() :])


def extract_labeled_value(
    lines: list[str],
    label_pattern: str,
    start: int = 0,
    end: int | None = None,
    multiline: bool = False,
    max_following_lines: int = 2,
) -> str | None:
    end = len(lines) if end is None else min(end, len(lines))
    for index in range(start, end):
        if is_provider_noise_line(lines[index]):
            continue
        cells = split_layout_cells(lines[index]) or [lines[index]]
        for cell_index, cell in enumerate(cells):
            if not re.search(label_pattern, cell, re.IGNORECASE):
                continue

            parts: list[str] = []
            same_cell_value = value_after_label_in_cell(cell, label_pattern)
            if same_cell_value:
                parts.append(same_cell_value)

            if not parts:
                for following_cell in cells[cell_index + 1 :]:
                    if is_known_label_cell(following_cell):
                        break
                    if following_cell and len(compact_match_text(following_cell)) > 2:
                        parts.append(following_cell)

            if multiline:
                for following_line in lines[index + 1 : min(end, index + 1 + max_following_lines)]:
                    if (
                        is_known_label_cell(following_line)
                        or is_table_header_line(following_line)
                        or is_table_total_boundary(following_line)
                    ):
                        break
                    parts.append(following_line)

            value = clean_labeled_value(" ".join(parts))
            if value:
                return value
    return None


def find_line_index(
    lines: list[str],
    *patterns: str,
    start: int = 0,
    end: int | None = None,
) -> int | None:
    end = len(lines) if end is None else min(end, len(lines))
    for index in range(start, end):
        if line_matches_compact(lines[index], *patterns):
            return index
    return None


def looks_like_company_name(value: Any) -> bool:
    compact = compact_match_text(value)
    return bool(
        re.search(
            r"congty|cty|trungtam|benhvien|chicuc|cuc|tongcongty|hopta",
            compact,
        )
    )


def extract_company_before_tax(lines: list[str], tax_index: int) -> str | None:
    candidates: list[str] = []
    for line in reversed(lines[max(0, tax_index - 5) : tax_index]):
        for cell in split_layout_cells(line) or [line]:
            if is_known_label_cell(cell) and not looks_like_company_name(cell):
                continue
            value = clean_labeled_value(cell)
            if value and looks_like_company_name(value):
                candidates.append(value)
    for value in candidates:
        if re.search(r"\bC[ÔO]NG\s*TY\b|\bCTY\b", value, re.IGNORECASE):
            return value
    return candidates[0] if candidates else None


def find_table_start_index(lines: list[str]) -> int | None:
    header_index = None
    for index, line in enumerate(lines):
        if is_table_header_line(line):
            header_index = index
            break

    if header_index is None:
        for index, line in enumerate(lines):
            if is_amount_column_marker(line):
                return index + 1
        return None

    start_index = header_index + 1
    for index in range(header_index + 1, min(len(lines), header_index + 12)):
        if is_table_header_guide_line(lines[index]):
            start_index = index + 1
    return start_index


def compact_table_marker(value: Any) -> str:
    value = clean_text_value(value) or ""
    return re.sub(r"\s+", "", value.lower()).replace("×", "x")


def is_amount_column_marker(value: Any) -> bool:
    return compact_table_marker(value) == "6=4x5"


def is_post_amount_header_marker(value: Any) -> bool:
    return compact_table_marker(value) in {"7", "8", "9=6+8"}


def is_table_total_boundary(value: Any) -> bool:
    value = clean_text_value(value) or ""
    compact = compact_match_text(value)
    return bool(
        re.search(
            r"congtien(?:hang|hanghoa|dichvu)|subtotal|tongcong|tongtien"
            r"|tongthanhtoan|sotien(?:viet)?bangchu|amountinwords|inwords",
            compact,
        )
    )


def is_item_line_number(value: Any) -> bool:
    value = clean_text_value(value) or ""
    return bool(re.fullmatch(r"\d{1,3}", value))


def is_numeric_cell(value: Any) -> bool:
    value = clean_text_value(value)
    if not value or "%" in value:
        return False
    value = re.sub(r"\b(?:vnd|vnđ|đ)\b", "", value, flags=re.IGNORECASE).strip()
    return bool(re.fullmatch(r"[-+]?\d+(?:[.,]\d+)*", value))


def numeric_token_matches(value: Any) -> list[re.Match[str]]:
    value = clean_text_value(value) or ""
    return list(re.finditer(r"[-+]?\d+(?:[.,]\d+)*(?:\s*%)?", value))


def has_numeric_token(value: Any) -> bool:
    return any("%" not in match.group(0) for match in numeric_token_matches(value))


def extract_numeric_tokens(value: Any, include_percent: bool = False) -> list[str]:
    tokens: list[str] = []
    for match in numeric_token_matches(value):
        token = re.sub(r"\s+", "", match.group(0))
        if "%" in token and not include_percent:
            continue
        tokens.append(token)
    return tokens


def is_percent_cell(value: Any) -> bool:
    value = clean_text_value(value) or ""
    return bool(re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?\s*%", value))


def numeric_cells_after_label(lines: list[str], pattern: str, count: int = 3) -> list[Any]:
    for index, line in enumerate(lines):
        if not re.search(pattern, line, re.IGNORECASE):
            continue
        values = []
        for following in lines[index + 1 :]:
            if is_numeric_cell(following):
                values.append(parse_vn_number(following))
                if len(values) >= count:
                    break
                continue
            if values:
                break
        return values
    return []


def text_after_colon(value: Any) -> str | None:
    value = clean_text_value(value)
    if not value or ":" not in value:
        return None
    return clean_text_value(value.split(":", 1)[1])


def split_contact_values(value: Any) -> list[str]:
    value = clean_text_value(value)
    if not value:
        return []
    return [item.strip() for item in re.split(r"\s+-\s+", value) if item.strip()]


def parse_seller_from_text(text: str) -> dict[str, Any]:
    lines = normalized_text_lines(text)
    seller = blank_invoice_json()["seller"]
    table_start = find_table_start_index(lines) or len(lines)
    seller_start = (
        find_line_index(lines, r"donvibanhang", r"seller", end=table_start) or 0
    )
    buyer_start = (
        find_line_index(
            lines,
            r"hovatennguoimuahang",
            r"hotennguoimuahang",
            r"nguoimuahang",
            r"customer",
            r"agent",
            start=seller_start,
            end=table_start,
        )
        or table_start
    )

    seller["name"] = extract_labeled_value(
        lines,
        r"Đơn\s*vị\s*bán\s*hàng\s*(?:\([^)]*\))?|Seller",
        start=seller_start,
        end=buyer_start,
    )
    seller["tax_code"] = normalize_tax_code(
        extract_labeled_value(
            lines,
            r"Mã\s*số\s*thuế\s*(?:\([^)]*\))?|MST|Tax\s*code|VAT\s*code",
            start=seller_start,
            end=buyer_start,
        )
    )
    if not seller["name"]:
        tax_index = find_line_index(
            lines,
            r"masothue",
            r"taxcode",
            r"vatcode",
            start=seller_start,
            end=buyer_start,
        )
        if tax_index is not None:
            seller["name"] = extract_company_before_tax(lines, tax_index)

    seller["address"] = extract_labeled_value(
        lines,
        r"Địa\s*chỉ\s*(?:\([^)]*\))?|Address",
        start=seller_start,
        end=buyer_start,
        multiline=True,
        max_following_lines=3,
    )
    phone = extract_labeled_value(
        lines,
        r"(?:ĐT|Điện\s*thoại|Tổng\s*đài|Tel\.?|Call\s*center)\s*(?:\([^)]*\))?",
        start=seller_start,
        end=buyer_start,
    )
    if phone:
        phone_part = re.split(r"\bFax\s*:?", phone, maxsplit=1, flags=re.IGNORECASE)[0]
        seller["phone"] = split_contact_values(phone_part)
    fax = extract_labeled_value(
        lines,
        r"Fax",
        start=seller_start,
        end=buyer_start,
    )
    if fax:
        seller["fax"] = split_contact_values(fax)
    seller["email"] = extract_labeled_value(
        lines,
        r"Email",
        start=seller_start,
        end=buyer_start,
    )
    seller["website"] = extract_labeled_value(
        lines,
        r"Website",
        start=seller_start,
        end=buyer_start,
    )

    for index, line in enumerate(lines[:buyer_start]):
        value = clean_text_value(line.strip("() "))
        if value and re.fullmatch(r"[A-Z0-9 .,&'/-]+", strip_accents(value)):
            previous = lines[index - 1] if index > 0 else ""
            if seller["name"] and value != seller["name"] and seller["name"] in previous:
                seller["english_name"] = value
                break

    return seller


def page_to_layout_text(page: Any, y_tolerance: float = 3.0, gap_tolerance: float = 12.0) -> str:
    words = page.get_text("words")
    if not words:
        return ""

    rows: list[dict[str, Any]] = []
    for word in sorted(words, key=lambda item: (round(item[1] / y_tolerance), item[0])):
        x0, y0, x1, y1, text, *_ = word
        center_y = (y0 + y1) / 2
        for row in rows:
            if abs(row["center_y"] - center_y) <= y_tolerance:
                row["words"].append((x0, y0, x1, y1, text))
                row["center_y"] = (
                    row["center_y"] * (len(row["words"]) - 1) + center_y
                ) / len(row["words"])
                break
        else:
            rows.append({"center_y": center_y, "words": [(x0, y0, x1, y1, text)]})

    lines: list[str] = []
    for row in sorted(rows, key=lambda item: item["center_y"]):
        parts: list[str] = []
        previous_x1 = None
        for x0, _y0, x1, _y1, text in sorted(row["words"], key=lambda item: item[0]):
            if previous_x1 is not None and x0 - previous_x1 > gap_tolerance:
                parts.append("|")
            parts.append(text)
            previous_x1 = x1
        line = clean_text_value(" ".join(parts))
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_pdf_text_pages(pdf_path: str | Path) -> list[str]:
    document = fitz.open(pdf_path)
    pages = []
    for page in document:
        layout_text = page_to_layout_text(page)
        raw_text = page.get_text("text")
        pages.append(layout_text or raw_text)
    document.close()
    return pages


COMMON_UNIT_VALUES = {
    "cai",
    "chiec",
    "bo",
    "hop",
    "thung",
    "goi",
    "chai",
    "lon",
    "kg",
    "g",
    "tan",
    "ta",
    "yen",
    "m",
    "m2",
    "m3",
    "cm",
    "cm2",
    "lan",
    "chuyen",
    "ngay",
    "gio",
    "thang",
    "pcs",
    "set",
    "unit",
    "dv",
    "dvt",
}


def split_description_unit(value: Any) -> tuple[str | None, str | None]:
    value = clean_text_value(value)
    if not value:
        return None, None
    parts = value.rsplit(" ", 1)
    if len(parts) == 2 and compact_match_text(parts[1]) in COMMON_UNIT_VALUES:
        return clean_text_value(parts[0]), clean_text_value(parts[1])
    return value, None


def is_table_noise_line(value: Any) -> bool:
    value = clean_text_value(value)
    if not value:
        return True
    compact = compact_match_text(value)
    if len(compact) <= 2:
        return True
    if is_provider_noise_line(value):
        return True
    if is_table_header_guide_line(value):
        return True
    return bool(
        re.search(
            r"chuyensangtrangsau|tieptheotrangtruoc|tracu|matracuu|trang\d+"
            r"|donvicungcapgiaiphap|giaiphaps|nguoimuahang|nguoibanhang"
            r"|signaturevalid|kyboi|kyngay",
            compact,
        )
    )


def parse_layout_item_row(line: str) -> dict[str, Any] | None:
    cells = split_layout_cells(line)
    if len(cells) < 2:
        return None

    line_number = None
    description = None
    unit = None
    numeric_start = None

    if is_item_line_number(cells[0]):
        line_number = int(clean_text_value(cells[0]) or 0)
        if len(cells) >= 5 and not is_numeric_cell(cells[1]):
            if has_numeric_token(cells[2]):
                unit = cells[1]
                numeric_start = 2
            elif len(cells) >= 6 and has_numeric_token(cells[3]):
                description = cells[1]
                unit = cells[2]
                numeric_start = 3
    else:
        match = re.match(r"^(\d{1,3})\s+(.+)$", cells[0])
        if match and len(cells) >= 4:
            line_number = int(match.group(1))
            description, unit = split_description_unit(match.group(2))
            numeric_start = 1

    if line_number is None or numeric_start is None:
        return None

    numeric_text = " ".join(cells[numeric_start:])
    number_tokens = extract_numeric_tokens(numeric_text)
    if len(number_tokens) < 3:
        return None

    percent_tokens = [
        token for token in extract_numeric_tokens(numeric_text, include_percent=True) if "%" in token
    ]
    quantity = parse_vn_number(number_tokens[0])
    unit_price = parse_vn_number(number_tokens[1])
    amount = parse_vn_number(number_tokens[2])
    total_amount = parse_vn_number(number_tokens[-1]) if len(number_tokens) >= 4 else None

    item = {
        "line_number": line_number,
        "description": description,
        "description_lines": [description] if description else [],
        "unit": clean_text_value(unit),
        "quantity": quantity,
        "unit_price": unit_price,
        "amount": amount,
        "taxable_amount": amount,
    }
    if percent_tokens:
        item["vat_rate"] = parse_vn_percent(percent_tokens[0])
        if len(number_tokens) >= 4:
            item["vat_amount"] = parse_vn_number(number_tokens[-2])
            item["total_amount"] = total_amount
    elif total_amount is not None:
        item["total_amount"] = total_amount

    return item


def normalize_description_lines(lines: list[str]) -> list[str]:
    normalized = []
    for line in lines:
        value = clean_text_value(line)
        if value and not is_table_noise_line(value):
            normalized.append(value)
    return normalized


def apply_item_description(
    item: dict[str, Any],
    prefix_lines: list[str],
    suffix_lines: list[str] | None = None,
) -> dict[str, Any]:
    description_lines = normalize_description_lines(prefix_lines)
    description_lines.extend(normalize_description_lines(item.get("description_lines", [])))
    if suffix_lines:
        description_lines.extend(normalize_description_lines(suffix_lines))
    if description_lines:
        item["description_lines"] = description_lines
        item["description"] = clean_text_value(" ".join(description_lines))
    return item


def parse_invoice_items_from_layout_lines(lines: list[str]) -> list[dict[str, Any]]:
    start_index = find_table_start_index(lines)
    if start_index is None:
        return []

    end_index = len(lines)
    for index in range(start_index, len(lines)):
        if is_table_total_boundary(lines[index]):
            end_index = index
            break

    table_lines = lines[start_index:end_index]
    row_infos: dict[int, dict[str, Any]] = {}
    for index, line in enumerate(table_lines):
        if is_table_noise_line(line):
            continue
        item = parse_layout_item_row(line)
        if item:
            row_infos[index] = item

    if not row_infos:
        return []

    items: list[dict[str, Any]] = []
    previous_after_row = 0
    row_indices = sorted(row_infos)
    for position, row_index in enumerate(row_indices):
        gap_lines = normalize_description_lines(table_lines[previous_after_row:row_index])
        prefix_lines: list[str] = []
        if items and gap_lines:
            if len(gap_lines) >= 2:
                previous_item = items[-1]
                apply_item_description(previous_item, [], gap_lines[:-1])
                prefix_lines = [gap_lines[-1]]
            elif row_infos[row_index].get("description"):
                previous_item = items[-1]
                apply_item_description(previous_item, [], gap_lines)
            else:
                prefix_lines = gap_lines
        else:
            prefix_lines = gap_lines

        item = apply_item_description(row_infos[row_index], prefix_lines)
        items.append(item)
        previous_after_row = row_index + 1

        if position == len(row_indices) - 1:
            trailing_lines = normalize_description_lines(table_lines[previous_after_row:end_index])
            if trailing_lines:
                apply_item_description(items[-1], [], trailing_lines)

    return [normalize_item(item) for item in items]


def parse_invoice_items_from_text(text: str) -> list[dict[str, Any]]:
    lines = normalized_text_lines(text)
    layout_items = parse_invoice_items_from_layout_lines(lines)
    if layout_items:
        return layout_items

    start_index = None
    end_index = len(lines)

    for index, line in enumerate(lines):
        if is_amount_column_marker(line):
            start_index = index + 1
            break

    if start_index is None:
        return []

    while start_index < len(lines) and is_post_amount_header_marker(lines[start_index]):
        start_index += 1

    for index in range(start_index, len(lines)):
        if is_table_total_boundary(lines[index]):
            end_index = index
            break

    table_lines = lines[start_index:end_index]
    items: list[dict[str, Any]] = []
    index = 0
    while index < len(table_lines):
        if not is_item_line_number(table_lines[index]):
            index += 1
            continue

        parsed_item = None
        next_index = index + 1
        for unit_index in range(index + 2, len(table_lines)):
            unit = table_lines[unit_index]
            if is_item_line_number(unit) or is_numeric_cell(unit) or is_percent_cell(unit):
                continue
            if unit_index + 3 >= len(table_lines):
                break
            if not (
                is_numeric_cell(table_lines[unit_index + 1])
                and is_numeric_cell(table_lines[unit_index + 2])
                and is_numeric_cell(table_lines[unit_index + 3])
            ):
                continue

            description_lines = table_lines[index + 1 : unit_index]
            if not description_lines:
                continue

            amount = parse_vn_number(table_lines[unit_index + 3])
            parsed_item = {
                "line_number": int(clean_text_value(table_lines[index]) or 0),
                "description": clean_text_value(" ".join(description_lines)),
                "description_lines": description_lines,
                "unit": clean_text_value(unit),
                "quantity": parse_vn_number(table_lines[unit_index + 1]),
                "unit_price": parse_vn_number(table_lines[unit_index + 2]),
                "amount": amount,
                "taxable_amount": amount,
            }
            next_index = unit_index + 4

            if (
                unit_index + 6 < len(table_lines)
                and is_percent_cell(table_lines[unit_index + 4])
                and is_numeric_cell(table_lines[unit_index + 5])
                and is_numeric_cell(table_lines[unit_index + 6])
            ):
                parsed_item.update(
                    {
                        "vat_rate": parse_vn_percent(table_lines[unit_index + 4]),
                        "vat_amount": parse_vn_number(table_lines[unit_index + 5]),
                        "total_amount": parse_vn_number(table_lines[unit_index + 6]),
                    }
                )
                next_index = unit_index + 7
            break

        if parsed_item:
            items.append(normalize_item(parsed_item))
            index = next_index
        else:
            index += 1

    return items


def extract_invoice_date_from_text(text: str) -> str | None:
    text = plain_layout_text(text)
    match = re.search(
        r"Ngày\s*(?:\([^)]*(?:Date|Day)[^)]*\))?\s*(\d{1,2})\s*"
        r"tháng\s*(?:\([^)]*Month[^)]*\))?\s*(\d{1,2})\s*"
        r"năm\s*(?:\([^)]*Year[^)]*\))?\s*(\d{4})",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    day, month, year = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def extract_tax_authority_code(lines: list[str]) -> str | None:
    value = extract_labeled_value(
        lines,
        r"Mã\s*(?:của\s*cơ\s*quan\s*thuế|CQT|CQ\s*thuế)\s*(?:\([^)]*\))?"
        r"|MCQT\s*cấp|MCCQT\s*(?:\([^)]*\))?",
        multiline=True,
        max_following_lines=1,
    )
    if not value:
        return None
    match = re.search(r"\b[A-Z0-9]{10,}\b", value, re.IGNORECASE)
    return match.group(0).upper() if match else clean_text_value(value)


def parse_buyer_from_lines(lines: list[str]) -> dict[str, Any]:
    buyer = blank_invoice_json()["buyer"]
    table_start = find_table_start_index(lines) or len(lines)
    buyer_start = find_line_index(
        lines,
        r"hovatennguoimuahang",
        r"hotennguoimuahang",
        r"nguoimuahang",
        r"customer",
        r"agent",
        start=0,
        end=table_start,
    )
    if buyer_start is None:
        return buyer

    person_name = extract_labeled_value(
        lines,
        r"(?:Họ\s*(?:và\s*)?tên\s*người\s*mua\s*hàng|Người\s*mua\s*hàng)\s*"
        r"(?:\([^)]*\))?|Customer(?:'s)?\s*name|Agent",
        start=buyer_start,
        end=table_start,
    )
    company_name = extract_labeled_value(
        lines,
        r"Tên\s*đơn\s*vị\s*(?:\([^)]*\))?|Company(?:'s)?\s*name",
        start=buyer_start,
        end=table_start,
        multiline=True,
        max_following_lines=1,
    )
    if company_name:
        buyer["company_name"] = company_name
        buyer["name"] = person_name if person_name and not looks_like_company_name(person_name) else None
    elif person_name and looks_like_company_name(person_name):
        buyer["company_name"] = person_name
    else:
        buyer["name"] = person_name

    buyer["tax_code"] = normalize_tax_code(
        extract_labeled_value(
            lines,
            r"Mã\s*số\s*thuế\s*(?:\([^)]*\))?|MST|Tax\s*code|VAT\s*code",
            start=buyer_start,
            end=table_start,
        )
    )
    buyer["address"] = extract_labeled_value(
        lines,
        r"Địa\s*chỉ\s*(?:\([^)]*\))?|Address",
        start=buyer_start,
        end=table_start,
        multiline=True,
        max_following_lines=3,
    )
    buyer["account_number"] = extract_labeled_value(
        lines,
        r"Số\s*tài\s*khoản\s*(?:\([^)]*\))?|Bank\s*A/C|A/C\s*No\.?|Account\s*No\.?",
        start=buyer_start,
        end=table_start,
    )
    buyer["id_card"] = extract_labeled_value(
        lines,
        r"CCCD\s*(?:\([^)]*\))?|ID\s*Card|Số\s*định\s*danh\s*cá\s*nhân",
        start=buyer_start,
        end=table_start,
    )
    buyer["budgetary_unit_code"] = extract_labeled_value(
        lines,
        r"Mã\s*ĐVQHNS\s*(?:\([^)]*\))?|Budgetary\s*Unit\s*Code|Mã\s*quan\s*hệ\s*ngân\s*sách",
        start=buyer_start,
        end=table_start,
    )
    buyer["passport_number"] = extract_labeled_value(
        lines,
        r"Số\s*hộ\s*chiếu\s*(?:\([^)]*\))?|PP\s*Number|Passport",
        start=buyer_start,
        end=table_start,
    )
    return buyer


def numbers_from_total_line(lines: list[str], matcher: Any, pick: str = "first") -> Any:
    for line in lines:
        if is_table_header_line(line) or is_table_header_guide_line(line):
            continue
        if not matcher(line):
            continue
        tokens = extract_numeric_tokens(line)
        if not tokens:
            continue
        token = tokens[-1] if pick == "last" else tokens[0]
        return parse_vn_number(token)
    return None


def percent_from_total_line(lines: list[str], matcher: Any) -> Any:
    for line in lines:
        if not matcher(line):
            continue
        percent_tokens = [
            token
            for token in extract_numeric_tokens(line, include_percent=True)
            if "%" in token
        ]
        if percent_tokens:
            return parse_vn_percent(percent_tokens[0])
        if "%" in line:
            number_tokens = extract_numeric_tokens(line)
            if number_tokens:
                return parse_vn_percent(number_tokens[0])
    return None


def parse_totals_from_lines(lines: list[str]) -> dict[str, Any]:
    totals = blank_invoice_json()["totals"]

    def is_subtotal_line(line: str) -> bool:
        compact = compact_match_text(line)
        return bool(
            (
                re.search(r"congtien(?:hang|hanghoa|dichvu)|subtotal|totalamount", compact)
                or compact.startswith("tongcong")
            )
            and "thanhtoan" not in compact
            and "vatamount" not in compact
            and "tienthue" not in compact
        )

    def is_total_payment_line(line: str) -> bool:
        compact = compact_match_text(line)
        return bool(
            re.search(
                r"tongcongtien(?:thanh)?toan|tongtien(?:thanh)?toan|"
                r"tongthanhtoan|totalpayment|total$",
                compact,
            )
        )

    def is_vat_rate_line(line: str) -> bool:
        return line_matches_compact(line, r"thue?su?a?tgtgt", r"vatrate")

    def is_vat_amount_line(line: str) -> bool:
        return line_matches_compact(line, r"ti?e?nthue?gtgt", r"vatamount", r"v?at$")

    subtotal_tokens: list[str] = []
    for line in lines:
        if is_table_header_line(line) or is_table_header_guide_line(line):
            continue
        if is_subtotal_line(line):
            subtotal_tokens = extract_numeric_tokens(line)
            if subtotal_tokens:
                totals["subtotal"] = parse_vn_number(subtotal_tokens[0])
                break
    totals["vat_rate"] = percent_from_total_line(lines, is_vat_rate_line)
    totals["vat_amount"] = numbers_from_total_line(lines, is_vat_amount_line, pick="last")
    totals["total_payment"] = numbers_from_total_line(lines, is_total_payment_line, pick="last")

    if len(subtotal_tokens) >= 3:
        if totals["vat_amount"] is None:
            totals["vat_amount"] = parse_vn_number(subtotal_tokens[-2])
        if totals["total_payment"] is None:
            totals["total_payment"] = parse_vn_number(subtotal_tokens[-1])

    if totals["total_payment"] is None and totals["subtotal"] is not None:
        vat_amount = totals["vat_amount"] or 0
        totals["total_payment"] = totals["subtotal"] + vat_amount

    for line in lines:
        compact = compact_match_text(line)
        if not re.search(r"sotien(?:viet)?bangchu|amountinwords|inwords", compact):
            continue
        value = re.sub(
            r"^.*?(?:Số\s*tiền(?:\s*viết)?\s*bằng\s*chữ|Amount\s*in\s*words|in\s*words)"
            r"\s*(?:\([^)]*\))?\s*:?",
            "",
            line,
            flags=re.IGNORECASE,
        )
        value = re.sub(r"^\s*\)?\s*:?\s*", "", value)
        value = re.sub(r"^\s*\|+\s*", "", value)
        value = re.sub(r"\s+:\s+", " ", value)
        totals["amount_in_words"] = clean_text_value(value)
        if totals["amount_in_words"]:
            break

    return totals


def extract_invoice_number_from_lines(lines: list[str], end: int | None = None) -> str | None:
    end = len(lines) if end is None else min(end, len(lines))
    label_pattern = (
        r"Số\s*(?!tài\s*khoản|điện\s*thoại|định\s*danh|hộ\s*chiếu|tiền)"
        r"(?:\([^)]*(?:Invoice\s*No\.?|Invoice\s*Number|No\.?|Number)[^)]*\))?"
        r"|Invoice\s*No\.?|Invoice\s*Number"
    )
    for line in lines[:end]:
        for cell_index, cell in enumerate(split_layout_cells(line) or [line]):
            compact = compact_match_text(cell)
            if (
                "maso" in compact
                or compact.startswith("mauso")
                or "formno" in compact
                or "taxcode" in compact
                or "vatcode" in compact
                or compact.startswith(("sotaikhoan", "sodienthoai", "sodinhdanh", "sohochieu"))
            ):
                continue
            if not (
                compact == "so"
                or compact.startswith(("sono", "soinvoice", "invoiceno", "invoicenumber"))
                or re.search(r"^so\d", compact)
            ):
                continue
            value = value_after_label_in_cell(cell, label_pattern)
            if not value:
                cells = split_layout_cells(line)
                if cell_index + 1 < len(cells) and not is_known_label_cell(cells[cell_index + 1]):
                    value = cells[cell_index + 1]
            value = clean_labeled_value(value)
            if value and re.search(r"\d", value):
                match = re.search(r"\b[A-Z0-9][A-Z0-9./-]*\b", value, re.IGNORECASE)
                return match.group(0) if match else value
    return None


def extract_invoice_series_from_lines(lines: list[str], end: int | None = None) -> str | None:
    end = len(lines) if end is None else min(end, len(lines))
    label_pattern = r"Ký\s*hiệu\s*(?:\([^)]*(?:Series|Serial)[^)]*\))?|Serial"
    for line in lines[:end]:
        for cell_index, cell in enumerate(split_layout_cells(line) or [line]):
            if not re.search(label_pattern, cell, re.IGNORECASE):
                continue
            value = value_after_label_in_cell(cell, label_pattern)
            if not value:
                cells = split_layout_cells(line)
                if cell_index + 1 < len(cells) and not is_known_label_cell(cells[cell_index + 1]):
                    value = cells[cell_index + 1]
            value = clean_labeled_value(value)
            if value:
                match = re.search(r"\b[A-Z0-9][A-Z0-9./-]*\b", value, re.IGNORECASE)
                return match.group(0) if match else value
    return None


def parse_invoice_text_layer(text: str) -> dict[str, Any]:
    data = blank_invoice_json()
    if not clean_text_value(text):
        return data
    lines = normalized_text_lines(text)
    table_start = find_table_start_index(lines) or len(lines)
    plain_text = "\n".join(plain_layout_text(line) for line in lines)
    compact_text = compact_match_text(plain_text)

    data["invoice"]["invoice_type"] = (
        "VAT_INVOICE"
        if (
            "hoadongiatrigiatang" in compact_text
            or "vatinvoice" in compact_text
            or ("hoadon" in compact_text and "giatrigiatang" in compact_text)
        )
        else None
    )
    data["invoice"]["invoice_number"] = extract_invoice_number_from_lines(
        lines, end=table_start
    )
    data["invoice"]["series"] = extract_invoice_series_from_lines(lines, end=table_start)
    data["invoice"]["invoice_date"] = extract_invoice_date_from_text(plain_text)
    data["invoice"]["tax_authority_code"] = extract_tax_authority_code(lines)
    data["invoice"]["currency"] = extract_labeled_value(
        lines,
        r"Đơn\s*vị\s*tiền\s*tệ\s*(?:\([^)]*\))?|Currency",
        end=table_start,
    )
    data["invoice"]["payment_method"] = extract_labeled_value(
        lines,
        r"Hình\s*thức\s*thanh\s*toán\s*(?:\([^)]*\))?|Payment\s*method|Mode\s*of\s*payment",
        end=table_start,
    )

    merge_known_fields(data["seller"], parse_seller_from_text(text))
    merge_known_fields(data["buyer"], parse_buyer_from_lines(lines))

    data["shipping"]["ship_from_warehouse"] = extract_labeled_value(
        lines,
        r"Kho\s*xuất\s*hàng\s*(?:\([^)]*\))?|Ship-from\s*warehouse",
    )
    data["items"] = parse_invoice_items_from_text(text)
    merge_known_fields(data["totals"], parse_totals_from_lines(lines))
    summary_amounts = numeric_cells_after_label(
        lines,
        r"^(?:Tổng cộng|Cộng tiền hàng).*Total amount",
        count=3,
    )
    if summary_amounts:
        if data["totals"]["subtotal"] is None:
            data["totals"]["subtotal"] = summary_amounts[0]
        if len(summary_amounts) >= 2 and data["totals"]["vat_amount"] is None:
            data["totals"]["vat_amount"] = summary_amounts[1]
        if len(summary_amounts) >= 3 and data["totals"]["total_payment"] is None:
            data["totals"]["total_payment"] = summary_amounts[2]
    if data["totals"]["vat_rate"] is None and data["items"]:
        item_vat_rates = [
            item["vat_rate"]
            for item in data["items"]
            if isinstance(item.get("vat_rate"), (int, float))
        ]
        if len(set(item_vat_rates)) == 1:
            data["totals"]["vat_rate"] = item_vat_rates[0]
    data["signature"]["is_valid"] = (
        True if re.search(r"Signature Valid", text, re.IGNORECASE) else None
    )
    for index, line in enumerate(lines):
        if not re.search(r"(?:Được\s*)?Ký\s*bởi|Signed\s*by", line, re.IGNORECASE):
            continue
        signed_by = extract_labeled_value(
            lines,
            r"(?:Được\s*)?Ký\s*bởi|Signed\s*by",
            start=index,
            end=min(len(lines), index + 3),
            multiline=True,
            max_following_lines=2,
        )
        if signed_by:
            data["signature"]["signed_by"] = signed_by
    for index, line in enumerate(lines):
        if not re.search(r"Ngày\s*ký|Ký\s*ngày|Signed\s*date", line, re.IGNORECASE):
            continue
        signed_date = extract_labeled_value(
            lines,
            r"Ngày\s*ký|Ký\s*ngày|Signed\s*date",
            start=index,
            end=min(len(lines), index + 2),
        )
        if signed_date:
            data["signature"]["signed_date"] = signed_date
    data["metadata"]["pdf_text_layer_used"] = True

    return normalize_invoice_data(data)


def has_meaningful_value(value: Any) -> bool:
    if value is None or value == "" or value == []:
        return False
    if isinstance(value, dict):
        return any(has_meaningful_value(item) for item in value.values())
    return True


def is_text_layer_data_sufficient(data: Any) -> bool:
    data = normalize_invoice_data(data)
    if not isinstance(data, dict) or "parse_error" in data:
        return False

    invoice = data.get("invoice", {})
    totals = data.get("totals", {})
    has_invoice_identity = has_meaningful_value(invoice.get("invoice_number")) or (
        has_meaningful_value(invoice.get("series"))
        and has_meaningful_value(invoice.get("invoice_date"))
    )
    has_line_details = has_meaningful_value(data.get("items"))
    has_total_details = has_meaningful_value(totals.get("total_payment")) or (
        has_meaningful_value(totals.get("subtotal"))
        and has_meaningful_value(totals.get("vat_amount"))
    )
    return has_invoice_identity and (has_line_details or has_total_details)


TEXT_LAYER_CAN_CLEAR = {
    ("invoice", "tax_authority_code"),
    ("buyer", "name"),
    ("buyer", "id_card"),
    ("buyer", "passport_number"),
    ("buyer", "account_number"),
    ("buyer", "budgetary_unit_code"),
    ("signature", "signed_date"),
}


def overlay_invoice_data(base: Any, overlay: Any) -> dict[str, Any]:
    base = normalize_invoice_data(base)
    overlay = normalize_invoice_data(overlay)
    if not isinstance(base, dict) or "parse_error" in base:
        base = blank_invoice_json()
    if not isinstance(overlay, dict) or "parse_error" in overlay:
        return base

    for section, overlay_value in overlay.items():
        if section == "items":
            if overlay_value:
                base["items"] = overlay_value
            continue

        if isinstance(overlay_value, dict) and isinstance(base.get(section), dict):
            for key, value in overlay_value.items():
                if has_meaningful_value(value) or (section, key) in TEXT_LAYER_CAN_CLEAR:
                    base[section][key] = value

    return base


def merge_invoice_page_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    merged = blank_invoice_json()
    merged["metadata"]["page_count"] = len(results)
    merged["metadata"]["ocr_processed"] = True

    items: list[dict[str, Any]] = []
    for result in results:
        page_data = normalize_invoice_data(result.get("data", {}))
        if not isinstance(page_data, dict) or "parse_error" in page_data:
            continue

        for section in ["invoice", "seller", "buyer", "shipping", "signature"]:
            source_section = page_data.get(section, {})
            if not isinstance(source_section, dict):
                continue
            for key, value in source_section.items():
                if has_meaningful_value(value) and not has_meaningful_value(
                    merged[section].get(key)
                ):
                    merged[section][key] = value

        source_totals = page_data.get("totals", {})
        if isinstance(source_totals, dict):
            for key, value in source_totals.items():
                if has_meaningful_value(value):
                    merged["totals"][key] = value

        if isinstance(page_data.get("items"), list):
            items.extend(
                normalize_item(item)
                for item in page_data["items"]
                if isinstance(item, dict)
            )

        source_metadata = page_data.get("metadata", {})
        if isinstance(source_metadata, dict):
            bool_metadata_keys = {
                "pdf_text_layer_used",
                "vision_model_used",
                "text_layer_shortcut_used",
                "text_layer_overlay_used",
                "blank_page_skipped",
            }
            for key in [
                "source_type",
                "pdf_text_layer_used",
                "vision_model_used",
                "pdf_text_strategy",
                "text_layer_shortcut_used",
                "text_layer_overlay_used",
                "blank_page_skipped",
            ]:
                value = source_metadata.get(key)
                if key in bool_metadata_keys:
                    merged["metadata"][key] = bool(merged["metadata"].get(key)) or bool(
                        value
                    )
                elif has_meaningful_value(value) and not has_meaningful_value(
                    merged["metadata"].get(key)
                ):
                    merged["metadata"][key] = value

    merged["items"] = items
    return normalize_invoice_data(merged)


def scan_invoice_file(
    input_path: str | Path,
    output_image_dir: str | Path | None = None,
    config: OCRConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or ocr_config_from_env()
    input_path = Path(input_path)
    is_pdf = input_path.suffix.lower() == ".pdf"

    images, source_type, image_paths = load_input_images(
        input_path,
        pdf_dpi=config.pdf_dpi,
        save_pdf_images=is_pdf,
        pdf_image_dir=output_image_dir,
    )

    pdf_text_pages: list[str] = []
    pdf_text_strategy = normalize_pdf_text_strategy(config.pdf_text_strategy)
    should_read_pdf_text = (
        config.use_pdf_text_layer and is_pdf and pdf_text_strategy != "off"
    )
    if should_read_pdf_text:
        pdf_text_pages = extract_pdf_text_pages(input_path)

    results: list[dict[str, Any]] = []
    for page_index, image in enumerate(images, start=1):
        raw_response = None
        text_layer_data = None
        if page_index - 1 < len(pdf_text_pages) and clean_text_value(
            pdf_text_pages[page_index - 1]
        ):
            text_layer_data = parse_invoice_text_layer(pdf_text_pages[page_index - 1])

        text_layer_shortcut_used = False
        text_layer_overlay_used = False
        if (
            pdf_text_strategy == "fast"
            and text_layer_data
            and is_text_layer_data_sufficient(text_layer_data)
        ):
            page_data = text_layer_data
            vision_model_used = False
            text_layer_shortcut_used = True
        elif is_pdf and not text_layer_data and is_probably_blank_image(image):
            page_data = blank_invoice_json()
            vision_model_used = False
        else:
            raw_response = run_vintern_server(image=image, config=config)
            page_data = normalize_invoice_data(parse_model_json(raw_response))
            if text_layer_data:
                page_data = overlay_invoice_data(page_data, text_layer_data)
                text_layer_overlay_used = True
            vision_model_used = True

        if not isinstance(page_data.get("metadata"), dict):
            page_data["metadata"] = {}
        page_data["metadata"].update(
            {
                "source_type": source_type,
                "page_count": len(images),
                "ocr_processed": True,
                "pdf_text_layer_used": bool(text_layer_data),
                "vision_model_used": vision_model_used,
                "pdf_text_strategy": pdf_text_strategy,
                "text_layer_shortcut_used": text_layer_shortcut_used,
                "text_layer_overlay_used": text_layer_overlay_used,
            }
        )
        if is_pdf and not text_layer_data and not vision_model_used:
            page_data["metadata"]["blank_page_skipped"] = True

        page_image_path = image_paths[page_index - 1]
        if page_image_path:
            page_data["metadata"]["page_image_path"] = str(page_image_path)

        result = {"page": page_index, "data": page_data}
        if config.include_raw_response and raw_response is not None:
            result["raw_response"] = raw_response
        results.append(result)

    return results


def estimate_quality_score(results: list[dict[str, Any]]) -> int:
    if not results:
        return 0

    data = results[0].get("data", {})
    checks = [
        data.get("invoice", {}).get("invoice_number"),
        data.get("invoice", {}).get("invoice_date"),
        data.get("seller", {}).get("name"),
        data.get("seller", {}).get("tax_code"),
        data.get("buyer", {}).get("company_name") or data.get("buyer", {}).get("name"),
        data.get("buyer", {}).get("tax_code"),
        data.get("items"),
        data.get("totals", {}).get("subtotal"),
        data.get("totals", {}).get("vat_amount"),
        data.get("totals", {}).get("total_payment"),
    ]
    filled = sum(1 for item in checks if has_meaningful_value(item))
    return min(99, max(70, round(78 + (filled / len(checks)) * 21)))
