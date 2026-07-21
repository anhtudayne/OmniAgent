"""
PDF Parser — parse PDF into text chunks + multimodal items.

Uses MinerU (magic_pdf) for high-quality extraction.  Falls back to
PyMuPDF if MinerU is not installed.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def parse_pdf(file_path: str, output_dir: str | None = None) -> List[Dict[str, Any]]:
    """Parse a PDF file and return a content list.

    Each element is a dict with at least a ``type`` key:
    - ``{"type": "text", "text": "..."}``
    - ``{"type": "image", "img_path": "...", "img_caption": [...]}``
    - ``{"type": "table", "table_body": "...", "table_caption": [...]}``

    Returns:
        Flat list of content dicts in reading order.
    """
    file_path = str(Path(file_path).resolve())
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    output_dir = output_dir or config.PARSER_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    try:
        return _parse_with_mineru(file_path, output_dir)
    except (FileNotFoundError, ImportError):
        logger.warning("MinerU not available, falling back to PyMuPDF")
        return _parse_with_pymupdf(file_path, output_dir)


def separate_content(
    content_list: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Separate a content list into pure text and multimodal items.

    Inspired by ``raganything.utils.separate_content``.

    Returns:
        (text_content, multimodal_items)
    """
    text_parts: List[str] = []
    multimodal_items: List[Dict[str, Any]] = []

    for idx, item in enumerate(content_list):
        content_type = item.get("type", "text")

        if content_type == "text":
            text = item.get("text", "")
            if text.strip():
                text_parts.append(text)
        else:
            # Keep a copy with index metadata
            mm_item = dict(item)
            mm_item["_index"] = idx
            multimodal_items.append(mm_item)

    text_content = "\n\n".join(text_parts)

    logger.info(
        "Separated content: %d chars text, %d multimodal items",
        len(text_content),
        len(multimodal_items),
    )
    return text_content, multimodal_items


# ──────────────────────────────────────────────────────────────────
# MinerU backend
# ──────────────────────────────────────────────────────────────────

def _parse_with_mineru(file_path: str, output_dir: str) -> List[Dict[str, Any]]:
    """Call MinerU CLI (``magic-pdf``) and read the content-list JSON."""
    # Check MinerU is installed
    import shutil
    if shutil.which("magic-pdf") is None:
        raise FileNotFoundError("magic-pdf CLI not found on PATH")

    file_stem = Path(file_path).stem
    cmd = [
        "magic-pdf",
        "-p", file_path,
        "-o", output_dir,
        "-m", "auto",
    ]

    logger.info("Running MinerU: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        raise RuntimeError(f"MinerU failed (rc={result.returncode}): {result.stderr[:500]}")

    # MinerU outputs content_list.json under <output_dir>/<stem>/auto/
    content_list_path = Path(output_dir) / file_stem / "auto" / "content_list.json"
    if not content_list_path.exists():
        raise FileNotFoundError(f"MinerU content_list.json not found at {content_list_path}")

    with open(content_list_path, "r", encoding="utf-8") as f:
        content_list = json.load(f)

    logger.info("MinerU parsed %d content items", len(content_list))
    return content_list


# ──────────────────────────────────────────────────────────────────
# PyMuPDF fallback
# ──────────────────────────────────────────────────────────────────

def _parse_with_pymupdf(file_path: str, output_dir: str) -> List[Dict[str, Any]]:
    """Simple fallback using PyMuPDF (fitz) to extract text and images."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "Neither MinerU nor PyMuPDF is installed. "
            "Install one: pip install pymupdf  OR  pip install magic-pdf"
        )

    doc = fitz.open(file_path)
    content_list: List[Dict[str, Any]] = []
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for page_num, page in enumerate(doc):
        # Extract text
        text = page.get_text("text")
        if text.strip():
            content_list.append({
                "type": "text",
                "text": text.strip(),
                "page_idx": page_num,
            })

        # Extract images
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:  # CMYK → RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                img_filename = f"page{page_num}_img{img_idx}.png"
                img_path = str(images_dir / img_filename)
                pix.save(img_path)

                content_list.append({
                    "type": "image",
                    "img_path": img_path,
                    "img_caption": [],
                    "page_idx": page_num,
                })
            except Exception as e:
                logger.warning("Failed to extract image on page %d: %s", page_num, e)

    doc.close()
    logger.info("PyMuPDF parsed %d content items", len(content_list))
    return content_list
