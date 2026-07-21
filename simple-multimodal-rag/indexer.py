"""
Indexer — describe multimodal content, chunk text, embed everything
into ChromaDB.

Core flow:
  PDF → parse → separate text / images / tables
  → describe images & tables via Gemini Vision
  → chunk all text (including descriptions)
  → embed chunks → store in ChromaDB
"""

from __future__ import annotations
 
import base64
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import google.generativeai as genai

import config
from parser import parse_pdf, separate_content

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Gemini setup
# ──────────────────────────────────────────────────────────────────

def _init_gemini():
    """Configure the Gemini SDK once."""
    if not config.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Set it via environment variable or in config.py"
        )
    genai.configure(api_key=config.GEMINI_API_KEY)


def _get_model():
    return genai.GenerativeModel(config.GEMINI_MODEL)


# ──────────────────────────────────────────────────────────────────
# Multimodal description
# ──────────────────────────────────────────────────────────────────

def describe_image(image_path: str) -> str:
    """Use Gemini Vision to generate a text description of an image.

    Returns a plain-text description string.
    """
    path = Path(image_path)
    if not path.exists():
        logger.warning("Image file not found: %s", image_path)
        return f"[Image not found: {path.name}]"

    model = _get_model()

    # Read image bytes
    image_bytes = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

    prompt = (
        "Describe this image in detail. Include all visible text, data, "
        "charts, diagrams, or key visual elements. Be specific and factual."
    )

    try:
        response = model.generate_content([
            prompt,
            {"mime_type": mime, "data": image_bytes},
        ])
        description = response.text.strip()
        logger.info("Described image %s (%d chars)", path.name, len(description))
        return description
    except Exception as e:
        logger.error("Failed to describe image %s: %s", image_path, e)
        return f"[Image description failed: {path.name}]"


def describe_table(table_body: str, caption: str = "") -> str:
    """Use Gemini to generate a text summary of a table.

    Args:
        table_body: Raw table content (HTML, Markdown, or plain text)
        caption: Optional table caption

    Returns a plain-text description/summary.
    """
    model = _get_model()

    prompt = (
        "Summarize the following table. Describe the key data, trends, "
        "and insights. Keep the summary concise but complete.\n\n"
    )
    if caption:
        prompt += f"Table caption: {caption}\n\n"
    prompt += f"Table content:\n{table_body}"

    try:
        response = model.generate_content(prompt)
        description = response.text.strip()
        logger.info("Described table (%d chars)", len(description))
        return description
    except Exception as e:
        logger.error("Failed to describe table: %s", e)
        return f"[Table: {caption or 'no caption'}]\n{table_body[:200]}"


# ──────────────────────────────────────────────────────────────────
# Text chunking
# ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Split text into overlapping chunks by approximate token count.

    Uses a simple heuristic: 1 token ≈ 4 characters.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    char_chunk = chunk_size * 4
    char_overlap = overlap * 4

    if len(text) <= char_chunk:
        return [text] if text.strip() else []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + char_chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += char_chunk - char_overlap

    return chunks


# ──────────────────────────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts using Gemini Embedding API.

    Returns list of embedding vectors.
    """
    if not texts:
        return []

    result = genai.embed_content(
        model=f"models/{config.GEMINI_EMBEDDING_MODEL}",
        content=texts,
        task_type="retrieval_document",
    )
    return result["embedding"]


def embed_query(text: str) -> List[float]:
    """Embed a single query text."""
    result = genai.embed_content(
        model=f"models/{config.GEMINI_EMBEDDING_MODEL}",
        content=text,
        task_type="retrieval_query",
    )
    return result["embedding"]


# ──────────────────────────────────────────────────────────────────
# ChromaDB storage
# ──────────────────────────────────────────────────────────────────

def _get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    client = chromadb.PersistentClient(path=config.CHROMADB_PATH)
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _make_id(text: str) -> str:
    """Generate a deterministic ID from text content."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────
# Main indexing pipeline
# ──────────────────────────────────────────────────────────────────

def index_document(file_path: str) -> Dict[str, Any]:
    """Index a PDF document into ChromaDB.

    Pipeline:
      1. Parse PDF → content list
      2. Separate text vs multimodal
      3. Describe images/tables with Gemini
      4. Chunk all text
      5. Embed → store in ChromaDB

    Returns:
        Stats dict with counts of indexed items.
    """
    _init_gemini()

    file_name = Path(file_path).name
    logger.info("Indexing document: %s", file_name)

    # Step 1: Parse
    content_list = parse_pdf(file_path)

    # Step 2: Separate
    text_content, multimodal_items = separate_content(content_list)

    # Step 3: Process multimodal → generate descriptions
    all_chunks: List[Dict[str, Any]] = []

    # 3a: Text chunks
    if text_content.strip():
        text_chunks = chunk_text(text_content)
        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "type": "text",
                    "source": file_name,
                    "chunk_index": i,
                },
            })
        logger.info("Created %d text chunks", len(text_chunks))

    # 3b: Image descriptions
    image_count = 0
    for item in multimodal_items:
        if item.get("type") != "image":
            continue

        img_path = item.get("img_path", "")
        captions = item.get("img_caption", [])
        caption_text = ", ".join(captions) if isinstance(captions, list) else str(captions)

        description = describe_image(img_path)

        # Create chunk: description + caption
        chunk_text_content = f"[Image: {caption_text}]\n{description}" if caption_text else description
        all_chunks.append({
            "text": chunk_text_content,
            "metadata": {
                "type": "image",
                "source": file_name,
                "img_path": img_path,
                "caption": caption_text,
                "page_idx": item.get("page_idx", -1),
            },
        })
        image_count += 1

    if image_count:
        logger.info("Described %d images", image_count)

    # 3c: Table descriptions
    table_count = 0
    for item in multimodal_items:
        if item.get("type") != "table":
            continue

        table_body = item.get("table_body", item.get("table_data", item.get("text", "")))
        captions = item.get("table_caption", [])
        caption_text = ", ".join(captions) if isinstance(captions, list) else str(captions)

        description = describe_table(str(table_body), caption_text)

        chunk_text_content = f"[Table: {caption_text}]\n{description}" if caption_text else description
        # Also append raw table data for precise retrieval
        chunk_text_content += f"\n\nRaw table data:\n{str(table_body)[:1000]}"

        all_chunks.append({
            "text": chunk_text_content,
            "metadata": {
                "type": "table",
                "source": file_name,
                "caption": caption_text,
                "page_idx": item.get("page_idx", -1),
            },
        })
        table_count += 1

    if table_count:
        logger.info("Described %d tables", table_count)

    if not all_chunks:
        logger.warning("No content extracted from %s", file_path)
        return {"text_chunks": 0, "images": 0, "tables": 0}

    # Step 4: Embed all chunks
    texts = [c["text"] for c in all_chunks]

    # Embed in batches (Gemini has a limit per request)
    BATCH_SIZE = 100
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        embeddings = embed_texts(batch)
        all_embeddings.extend(embeddings)
        logger.info("Embedded batch %d/%d", i // BATCH_SIZE + 1, (len(texts) - 1) // BATCH_SIZE + 1)

    # Step 5: Store in ChromaDB
    collection = _get_collection()

    ids = [_make_id(t) for t in texts]
    metadatas = [c["metadata"] for c in all_chunks]

    # ChromaDB expects metadata values to be str, int, float, or bool
    clean_metadatas = []
    for m in metadatas:
        clean = {}
        for k, v in m.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        clean_metadatas.append(clean)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=all_embeddings,
        metadatas=clean_metadatas,
    )

    stats = {
        "total_chunks": len(all_chunks),
        "text_chunks": len([c for c in all_chunks if c["metadata"]["type"] == "text"]),
        "images": image_count,
        "tables": table_count,
    }

    logger.info(
        "Indexing complete: %d total chunks (%d text, %d images, %d tables)",
        stats["total_chunks"], stats["text_chunks"], stats["images"], stats["tables"],
    )
    return stats
