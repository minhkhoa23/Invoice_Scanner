"""
Local CPU invoice OCR pipeline using the quantized GGUF Vintern model.

Examples:
    python ocr_invoice_gguf_local.py --input "invoice.pdf" --output "invoice.json"
    python ocr_invoice_gguf_local.py --input "invoice.jpg" --output "invoice.json"

The first run downloads the selected GGUF file from Hugging Face. Inference is
CPU-only by default via n_gpu_layers=0, so it can run without a discrete GPU.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any


MODEL_REPO_ID = "rootonchair/Vintern-1B-v3_5-GGUF-ext"

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
            "  pip install -r requirements-gguf-local.txt"
        ) from error

    model_filename = get_quant_filename(quant=quant, filename=filename)

    return Llama.from_pretrained(
        repo_id=repo_id,
        filename=model_filename,
        n_ctx=n_ctx,
        n_threads=n_threads or max(1, (os.cpu_count() or 4) - 1),
        n_gpu_layers=0,
        verbose=verbose,
    )


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


def response_to_text(response: Any) -> str:
    if isinstance(response, dict):
        choices = response.get("choices")
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content

    return str(response)


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

    response = llm.create_chat_completion(
        messages=[
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
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return response_to_text(response)


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

    for page_index, image in enumerate(images, start=1):
        print(f"Processing page {page_index}/{len(images)}...")

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

        results.append(
            {
                "page": page_index,
                "data": page_data,
                "raw_response": raw_response,
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

    llm = load_llm(
        repo_id=args.repo_id,
        quant=args.quant,
        filename=args.filename,
        n_ctx=args.n_ctx,
        n_threads=args.threads,
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

    save_results(
        results=results,
        output_path=output_path,
        include_raw_response=args.include_raw_response,
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
