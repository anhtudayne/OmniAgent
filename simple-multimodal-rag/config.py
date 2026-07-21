"""
Configuration for Simple Multimodal RAG System
"""

import os
from pathlib import Path

# ─── Google Gemini ────────────────────────────────────────────────
GEMINI_API_KEY = "your_api_key_here"  # Replace with your actual Gemini API key

# Model cho text generation + vision (mô tả ảnh/bảng + trả lời query)
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Model cho embedding
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

# ─── ChromaDB ─────────────────────────────────────────────────────
CHROMADB_PATH = os.getenv(
    "CHROMADB_PATH",
    str(Path(__file__).parent / "chroma_db"),
)
COLLECTION_NAME = "multimodal_rag"

# ─── Chunking ─────────────────────────────────────────────────────
CHUNK_SIZE = 512        # tokens per chunk (approximate by chars / 4)
CHUNK_OVERLAP = 64      # overlap tokens between chunks

# ─── Retrieval ────────────────────────────────────────────────────
TOP_K = 5               # number of chunks to retrieve per query

# ─── Parser ───────────────────────────────────────────────────────
PARSER_OUTPUT_DIR = str(Path(__file__).parent / "parser_output")
