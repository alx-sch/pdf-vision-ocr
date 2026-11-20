#!/usr/bin/env python3
"""Extract text from PDFs using any OpenAI-compatible vision API."""

import argparse
import base64
import json
import os
import re
import ssl
import sys
import time
from pathlib import Path

import httpx
import pymupdf as fitz

DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 4096
RENDER_SCALE = 2.0

ORIENTATION_PROMPT = (
    "Is this page rotated or upside down? Reply with ONLY one of: "
    "0, 90, 180, 270 — the degrees of clockwise rotation needed to make "
    "the text upright. If the page is already upright, reply 0."
)

OCR_PROMPT = (
    "Extract all text from this image exactly as it appears. "
    "If the page appears rotated or upside down, read it in its correct orientation. "
    "Preserve the original layout as closely as possible. "
    "If an area is visibly redacted or blacked out, write [REDACTED] in its place.\n\n"
    "If the image contains a table: after the extracted text, add a section "
    "headed [TABLE] that reproduces the table as a markdown table. "
    "Flatten any nested or multi-level column headers into single-row headers "
    "that carry full context (e.g. if 'Men' and 'Women' are sub-columns under "
    "'NT-proBNP (pg/mL)', use 'NT-proBNP Men (pg/mL)' and "
    "'NT-proBNP Women (pg/mL)' as column names). "
    "Every cell value must be unambiguously tied to its row and column.\n\n"
    "If the image contains a diagram, flowchart, decision tree, or figure: "
    "after the extracted text, add a section headed [DIAGRAM] that describes "
    "the structure — nodes, connections, and decision logic — so a reader "
    "who cannot see the image understands the relationships.\n\n"
    "If the image contains a clinical recommendation box or graded panel "
    "(e.g. Class I/IIa/IIb/III, Level of Evidence A/B/C): after the extracted "
    "text, add a section headed [RECOMMENDATION] that states the recommendation "
    "text together with its class and level of evidence.\n\n"
    "If the image contains a forest plot or statistical figure: after the "
    "extracted text, add a section headed [FIGURE] that describes what the "
    "figure shows — comparisons, direction of effect, and key numeric results.\n\n"
    "A page may contain multiple tables, diagrams, recommendations, and/or "
    "figures — include the appropriate section for each one.\n\n"
    "Do not add any other commentary or preamble."
)

HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*[^>]*>")


def clean_text(text: str) -> str:
    text = HTML_TAG_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def progress_bar(current: int, total: int, width: int = 30) -> str:
    filled = int(width * current / total) if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def build_client(args: argparse.Namespace) -> httpx.Client:
    kwargs: dict = {"timeout": 300.0}
    if args.ca_cert:
        ctx = ssl.create_default_context()
        ctx.load_verify_locations(args.ca_cert)
        kwargs["verify"] = ctx
    return httpx.Client(**kwargs)


def _api_call(
    client: httpx.Client,
    api_base: str,
    api_key: str,
    model: str,
    img_bytes: bytes,
    prompt: str,
    max_tokens: int,
) -> str:
    b64 = base64.b64encode(img_bytes).decode()
    endpoint = f"{api_base.rstrip('/')}/chat/completions"
    resp = client.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def vision_call(
    client: httpx.Client,
    api_base: str,
    api_key: str,
    model: str,
    img_bytes: bytes,
    max_tokens: int,
) -> str:
    return _api_call(client, api_base, api_key, model, img_bytes, OCR_PROMPT, max_tokens)


def detect_orientation(
    client: httpx.Client,
    api_base: str,
    api_key: str,
    model: str,
    img_bytes: bytes,
) -> int:
    resp = _api_call(
        client, api_base, api_key, model, img_bytes, ORIENTATION_PROMPT, 16
    )
    for token in resp.strip().split():
        if token in ("0", "90", "180", "270"):
            return int(token)
    return 0


def pdf_page_to_png(data: bytes, page_idx: int, rotation: int = 0) -> bytes:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        page = doc.load_page(page_idx)
        mat = fitz.Matrix(RENDER_SCALE, RENDER_SCALE).prerotate(rotation)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")
    finally:
        doc.close()


def pdf_page_count(filepath: Path) -> int:
    doc = fitz.open(str(filepath))
    n = len(doc)
    doc.close()
    return n


def log(msg: str, log_file: Path | None = None) -> None:
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def extract_pdf(
    filepath: Path,
    client: httpx.Client,
    args: argparse.Namespace,
    *,
    pages_done_global: int,
    total_pages_global: int,
    batch_start: float,
) -> tuple[str, int]:
    data = filepath.read_bytes()
    total_pages = pdf_page_count(filepath)

    parts: list[str] = []
    for i in range(total_pages):
        page_num = i + 1
        current_global = pages_done_global + page_num
        pct = current_global / total_pages_global * 100

        elapsed_total = time.time() - batch_start
        avg_per_page = elapsed_total / current_global if current_global else 0
        remaining = (total_pages_global - current_global) * avg_per_page
        eta_str = fmt_duration(remaining)

        bar = progress_bar(current_global, total_pages_global)
        status = (
            f"\r  {bar} {current_global}/{total_pages_global} pages "
            f"({pct:.0f}%)  |  page {page_num}/{total_pages}  |  ETA {eta_str}   "
        )
        print(status, end="", flush=True)

        rotation = 0
        if args.fix_orientation:
            low_res_png = pdf_page_to_png(data, i)
            rotation = detect_orientation(
                client, args.api_base, args.api_key, args.model, low_res_png
            )
            if rotation:
                print(f" (rotated {rotation}°)", end="", flush=True)

        png = pdf_page_to_png(data, i, rotation=rotation)
        text = vision_call(
            client, args.api_base, args.api_key, args.model, png, args.max_tokens
        ).strip()
        if text:
            parts.append(f"--- Page {page_num} ---\n{text}")

    print()
    return clean_text("\n\n".join(parts)), total_pages


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract text from PDFs using a vision LLM (OpenAI-compatible API).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # OpenAI
  %(prog)s --api-key $OPENAI_API_KEY docs/*.pdf

  # Azure OpenAI
  %(prog)s --api-base https://my-resource.openai.azure.com/openai/deployments/gpt-4o \\
           --api-key $AZURE_KEY docs/*.pdf

  # Any OpenAI-compatible endpoint
  %(prog)s --api-base https://my-gateway.com/v1 --api-key $KEY --model gpt-4o docs/*.pdf

  # Custom output directory
  %(prog)s --api-key $KEY -o extracted/ docs/*.pdf
""",
    )
    p.add_argument("files", nargs="+", type=Path, help="PDF files to extract")
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output directory (default: <input-dir>/extracted/)",
    )
    p.add_argument(
        "--api-base", default=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        help="API base URL (default: $OPENAI_API_BASE or https://api.openai.com/v1)",
    )
    p.add_argument(
        "--api-key", default=os.getenv("OPENAI_API_KEY", ""),
        help="API key (default: $OPENAI_API_KEY)",
    )
    p.add_argument(
        "--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"Vision model name (default: $OPENAI_MODEL or {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per page response (default: {DEFAULT_MAX_TOKENS})",
    )
    p.add_argument(
        "--ca-cert", default=os.getenv("SSL_CA_CERT"),
        help="Path to custom CA certificate bundle (default: $SSL_CA_CERT)",
    )
    p.add_argument(
        "--fix-orientation", action="store_true",
        help="Detect and correct rotated/upside-down pages (extra API call per page)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-extract even if output file already exists",
    )

    args = p.parse_args()

    if not args.api_key:
        p.error("API key required: set --api-key or $OPENAI_API_KEY")

    for f in args.files:
        if not f.exists():
            p.error(f"File not found: {f}")
        if f.suffix.lower() != ".pdf":
            p.error(f"Not a PDF: {f}")

    if args.output is None:
        args.output = args.files[0].parent / "extracted"

    return args


def main() -> None:
    args = parse_args()
    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "_progress.log"

    page_counts = {pdf: pdf_page_count(pdf) for pdf in args.files}

    if args.force:
        to_extract = list(page_counts.items())
    else:
        to_extract = [(pdf, pc) for pdf, pc in page_counts.items()
                      if not (out_dir / f"{pdf.stem}.txt").exists()]
    skipped = len(args.files) - len(to_extract)
    total_pages = sum(pc for _, pc in to_extract)

    header = (
        f"{'=' * 60}\n"
        f"  pdf-vision-ocr — {args.model} via {args.api_base}\n"
        f"{'=' * 60}\n"
        f"  PDFs to extract:  {len(to_extract)}"
        + (f"  ({skipped} already done)" if skipped else "") + "\n"
        f"  Total pages:      {total_pages}\n"
        f"  Est. time:        {fmt_duration(total_pages * 18)}\n"
        f"  Output:           {out_dir}/\n"
        f"  Progress log:     {log_file}\n"
        f"{'=' * 60}"
    )
    log(header, log_file)

    if not to_extract:
        log("Nothing to do — all PDFs already extracted.", log_file)
        return

    pages_done = 0
    batch_start = time.time()

    client = build_client(args)
    try:
        for idx, (pdf, pc) in enumerate(to_extract, 1):
            log(f"\n[{idx}/{len(to_extract)}] {pdf.name}  ({pc} pages)", log_file)
            try:
                text, extracted_pages = extract_pdf(
                    pdf, client, args,
                    pages_done_global=pages_done,
                    total_pages_global=total_pages,
                    batch_start=batch_start,
                )
                pages_done += extracted_pages
                out_path = out_dir / f"{pdf.stem}.txt"
                out_path.write_text(text, encoding="utf-8")
                elapsed_so_far = time.time() - batch_start
                log(
                    f"  -> {out_path.name}  ({len(text):,} chars, {fmt_duration(elapsed_so_far)} elapsed)",
                    log_file,
                )
            except Exception as e:
                pages_done += pc
                log(f"  ERROR: {e}", log_file)
    finally:
        client.close()

    elapsed = time.time() - batch_start
    summary = (
        f"\n{'=' * 60}\n"
        f"  Done!  {pages_done} pages in {fmt_duration(elapsed)}\n"
        f"  Results in {out_dir}/\n"
        f"{'=' * 60}"
    )
    log(summary, log_file)


if __name__ == "__main__":
    main()
