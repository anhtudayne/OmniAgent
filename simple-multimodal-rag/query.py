"""
Query — retrieve relevant chunks from ChromaDB and generate
an answer using Gemini.

Flow:
  User question → embed → find top-k similar chunks in ChromaDB
  → build context → Gemini generates answer
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import google.generativeai as genai

import config
from indexer import _get_collection, _init_gemini, embed_query

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────

def retrieve(question: str, top_k: int = None) -> List[Dict[str, Any]]:
    """Retrieve the most relevant chunks for a question.

    Returns a list of dicts with keys: text, metadata, distance.
    """
    _init_gemini()

    top_k = top_k or config.TOP_K
    collection = _get_collection()

    if collection.count() == 0:
        logger.warning("ChromaDB collection is empty — index a document first.")
        return []

    q_embedding = embed_query(question)

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "metadata": meta,
            "distance": dist,
        })

    logger.info("Retrieved %d chunks (top distance: %.4f)", len(chunks), chunks[0]["distance"] if chunks else 0)
    return chunks


# ──────────────────────────────────────────────────────────────────
# Answer generation
# ──────────────────────────────────────────────────────────────────

def _build_context(chunks: List[Dict[str, Any]]) -> str:
    """Build a context string from retrieved chunks."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        content_type = chunk["metadata"].get("type", "text")
        header = f"[Chunk {i} | type={content_type} | source={source}]"
        parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "The context may include text passages, image descriptions, and table summaries "
    "extracted from documents. Use only the provided context to answer. "
    "If the context does not contain enough information, say so clearly. "
    "Always cite which chunk(s) your answer is based on."
)


def query(question: str, top_k: int = None, system_prompt: str = None) -> str:
    """End-to-end query: retrieve context and generate an answer.

    Args:
        question: User's question
        top_k: Number of chunks to retrieve (default from config)
        system_prompt: Optional custom system prompt

    Returns:
        Answer string from Gemini.
    """
    _init_gemini()

    # 1. Retrieve
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return "No relevant information found. Please index a document first."

    # 2. Build context
    context = _build_context(chunks)

    # 3. Generate answer
    model = genai.GenerativeModel(
        config.GEMINI_MODEL,
        system_instruction=system_prompt or SYSTEM_PROMPT,
    )

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer based on the context above:"
    )

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        logger.info("Generated answer (%d chars)", len(answer))
        return answer
    except Exception as e:
        logger.error("Failed to generate answer: %s", e)
        return f"Error generating answer: {e}"


def query_with_sources(question: str, top_k: int = None) -> Dict[str, Any]:
    """Query and return both the answer and the source chunks.

    Returns:
        Dict with keys: answer, sources
    """
    _init_gemini()

    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "No relevant information found. Please index a document first.",
            "sources": [],
        }

    context = _build_context(chunks)
    model = genai.GenerativeModel(
        config.GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer based on the context above:"
    )

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Error generating answer: {e}"

    return {
        "answer": answer,
        "sources": [
            {
                "text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                "type": c["metadata"].get("type", "text"),
                "source": c["metadata"].get("source", "unknown"),
                "distance": round(c["distance"], 4),
            }
            for c in chunks
        ],
    }
