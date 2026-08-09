from __future__ import annotations

import ast
import base64
import json
import os
import re
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
            "unit": "string | null",
            "quantity": "number | null",
            "unit_price": "number | null",
            "amount": "number | null",
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
    include_raw_response: bool = False


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


def normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


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
        else:
            target[key] = value


def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    template = {
        "line_number": None,
        "description": None,
        "unit": None,
        "quantity": None,
        "unit_price": None,
        "amount": None,
    }
    merge_known_fields(template, item)
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
    value = re.sub(r"[ \t]+", " ", str(value))
    value = re.sub(r"\s*\n\s*", " ", value)
    value = value.strip(" :-")
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


def extract_pdf_text_pages(pdf_path: str | Path) -> list[str]:
    document = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in document]
    document.close()
    return pages


def parse_invoice_items_from_text(text: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    start_index = None
    end_index = len(lines)

    for index, line in enumerate(lines):
        if "6 = 4 x 5" in line:
            start_index = index + 1
            break

    if start_index is None:
        return []

    for index in range(start_index, len(lines)):
        if "Cộng tiền hàng" in lines[index] or "Total amount" in lines[index]:
            end_index = index
            break

    table_lines = lines[start_index:end_index]
    items = []
    index = 0
    while index + 5 < len(table_lines):
        if not re.fullmatch(r"\d+", table_lines[index]):
            index += 1
            continue

        unit = table_lines[index + 2]
        if not re.fullmatch(r"[A-Za-zÀ-ỹ/%\.]+", unit):
            index += 1
            continue

        items.append(
            {
                "line_number": int(table_lines[index]),
                "description": clean_text_value(table_lines[index + 1]),
                "unit": clean_text_value(unit),
                "quantity": parse_vn_number(table_lines[index + 3]),
                "unit_price": parse_vn_number(table_lines[index + 4]),
                "amount": parse_vn_number(table_lines[index + 5]),
            }
        )
        index += 6

    return items


def parse_invoice_text_layer(text: str) -> dict[str, Any]:
    data = blank_invoice_json()
    if not clean_text_value(text):
        return data

    data["invoice"]["invoice_type"] = (
        "VAT_INVOICE"
        if re.search(
            r"H[ÓO]A ĐƠN GI[ÁA] TRỊ GIA TĂNG|VAT INVOICE",
            text,
            re.IGNORECASE,
        )
        else None
    )
    data["invoice"]["invoice_number"] = regex_value(r"Số \(No\):[ \t]*([^\n]+)", text)
    data["invoice"]["series"] = regex_value(r"Ký hiệu \(Series\):[ \t]*([^\n]+)", text)
    day = regex_value(r"Ngày \(Date\)[ \t]*(\d{1,2})", text)
    month = regex_value(r"Tháng \(Month\)[ \t]*(\d{1,2})", text)
    year = regex_value(r"Năm \(Year\)[ \t]*(\d{4})", text)
    if day and month and year:
        data["invoice"]["invoice_date"] = f"{year}-{int(month):02d}-{int(day):02d}"
    data["invoice"]["tax_authority_code"] = regex_value(
        r"Mã CQT:[ \t]*([^\n]*)",
        text,
        flags=re.IGNORECASE,
    )
    data["invoice"]["payment_method"] = regex_value(
        r"Hình thức thanh toán \(Method payment\):[ \t]*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )

    seller_match = re.search(
        r"(?m)^([^\n:]+)\n([^\n:]+)\nMã số thuế \(Tax code\):\s*([^\n]+)\n"
        r"Địa chỉ \(Address\):\s*(.*?)\nĐT \(Tel\):\s*(.*?)\nEmail:\s*(.*?)\s+Website:\s*([^\n]+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if seller_match:
        data["seller"].update(
            {
                "name": clean_text_value(seller_match.group(1)),
                "english_name": clean_text_value(seller_match.group(2)),
                "tax_code": normalize_tax_code(seller_match.group(3)),
                "address": clean_text_value(seller_match.group(4)),
                "email": clean_text_value(seller_match.group(6)),
                "website": clean_text_value(seller_match.group(7)),
            }
        )
        contact_line = clean_text_value(seller_match.group(5)) or ""
        phone_part = contact_line.split("Fax:", 1)[0]
        fax_part = contact_line.split("Fax:", 1)[1] if "Fax:" in contact_line else ""
        data["seller"]["phone"] = [
            item.strip() for item in re.split(r"\s+-\s+", phone_part) if item.strip()
        ]
        data["seller"]["fax"] = [
            item.strip() for item in re.split(r"\s+-\s+", fax_part) if item.strip()
        ]

    data["buyer"]["name"] = regex_value(
        r"Họ tên người mua hàng \(Buyer's name\):[ \t]*([^\n]*)",
        text,
        flags=re.IGNORECASE,
    )
    data["buyer"]["company_name"] = regex_value(
        r"Tên đơn vị \(Company's name\):\s*(.*?)\n\s*Mã số thuế",
        text,
    )
    data["buyer"]["tax_code"] = normalize_tax_code(
        regex_value(
            r"Tên đơn vị \(Company's name\):.*?\n\s*Mã số thuế \(Tax code\):\s*([^\n]+)",
            text,
        )
    )
    data["buyer"]["address"] = regex_value(
        r"Địa chỉ:\s*\(Address\):\s*(.*?)\n\s*Hình thức thanh toán",
        text,
    )
    data["buyer"]["id_card"] = regex_value(
        r"CCCD \(ID Card\):[ \t]*([^\n]*)",
        text,
        flags=re.IGNORECASE,
    )
    data["buyer"]["account_number"] = regex_value(
        r"Số tài khoản \(Account number\):[ \t]*([^\n]*)",
        text,
        flags=re.IGNORECASE,
    )
    data["buyer"]["budgetary_unit_code"] = regex_value(
        r"Mã ĐVQHNS \(Budgetary Unit Code\):[ \t]*([^\n]*)",
        text,
        flags=re.IGNORECASE,
    )
    data["buyer"]["passport_number"] = regex_value(
        r"Số hộ chiếu \(PP Number\):[ \t]*([^\n]*)",
        text,
        flags=re.IGNORECASE,
    )
    data["shipping"]["ship_from_warehouse"] = regex_value(
        r"Kho xuất hàng \(Ship-from warehouse\):\s*(.*?)\n\s*Mã ĐVQHNS",
        text,
    )
    data["items"] = parse_invoice_items_from_text(text)
    data["totals"]["subtotal"] = parse_vn_number(
        regex_value(r"Cộng tiền hàng \(Total amount\):\s*([^\n]+)", text)
    )
    data["totals"]["vat_rate"] = parse_vn_percent(
        regex_value(r"Thuế suất GTGT \(VAT rate\):\s*([^\n]+)", text)
    )
    data["totals"]["vat_amount"] = parse_vn_number(
        regex_value(r"Tiền thuế GTGT \(VAT amount\):\s*([^\n]+)", text)
    )
    data["totals"]["total_payment"] = parse_vn_number(
        regex_value(r"Tổng cộng tiền thanh toán \(Total payment\):\s*([^\n]+)", text)
    )
    data["totals"]["amount_in_words"] = regex_value(
        r"Số tiền viết bằng chữ \(Amount in words\):\s*([^\n]+)",
        text,
    )
    data["signature"]["is_valid"] = (
        True if re.search(r"Signature Valid", text, re.IGNORECASE) else None
    )
    data["signature"]["signed_by"] = regex_value(r"Ký bởi:\s*([^\n]+)", text)
    data["signature"]["signed_date"] = regex_value(
        r"Ngày ký:[ \t]*([^\n]*)",
        text,
        flags=re.IGNORECASE,
    )
    data["metadata"]["pdf_text_layer_used"] = True

    return normalize_invoice_data(data)


def has_meaningful_value(value: Any) -> bool:
    if value is None or value == "" or value == []:
        return False
    if isinstance(value, dict):
        return any(has_meaningful_value(item) for item in value.values())
    return True


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
    if config.use_pdf_text_layer and is_pdf:
        pdf_text_pages = extract_pdf_text_pages(input_path)

    results: list[dict[str, Any]] = []
    for page_index, image in enumerate(images, start=1):
        raw_response = run_vintern_server(image=image, config=config)
        page_data = normalize_invoice_data(parse_model_json(raw_response))

        if page_index - 1 < len(pdf_text_pages) and clean_text_value(
            pdf_text_pages[page_index - 1]
        ):
            text_layer_data = parse_invoice_text_layer(pdf_text_pages[page_index - 1])
            page_data = overlay_invoice_data(page_data, text_layer_data)

        if not isinstance(page_data.get("metadata"), dict):
            page_data["metadata"] = {}
        page_data["metadata"].update(
            {
                "source_type": source_type,
                "page_count": len(images),
                "ocr_processed": True,
            }
        )

        page_image_path = image_paths[page_index - 1]
        if page_image_path:
            page_data["metadata"]["page_image_path"] = str(page_image_path)

        result = {"page": page_index, "data": page_data}
        if config.include_raw_response:
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
