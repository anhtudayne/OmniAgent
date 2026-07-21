import json
import os
import re
from pathlib import Path

import numpy as np
import faiss
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[4]
dotenv_path = PROJECT_ROOT / "src" / "config" / ".env"
load_dotenv(dotenv_path)

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "and", "but", "or", "nor", "not", "so",
    "yet", "both", "either", "neither", "each", "every", "all", "any",
    "few", "more", "most", "other", "some", "such", "no", "only", "own",
    "same", "than", "too", "very", "just", "because", "if", "when",
    "where", "how", "what", "which", "who", "whom", "this", "that",
    "these", "those", "it", "its",
})


def tokenize_for_bm25(text):
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", text)
    result = []
    for t in tokens:
        if t in _STOP_WORDS:
            continue
        if len(t) >= 2 or t.isdigit():
            result.append(t)
    return result


class RAGPipeline:
    def __init__(self, chunks_path=None, faiss_index_path=None, embedding_model_name=None):
        self.chunks_path = Path(chunks_path or (PROJECT_ROOT / "data" / "reading_parameters" / "chunks_action.jsonl"))
        self.faiss_index_path = Path(faiss_index_path or (PROJECT_ROOT / "data" / "reading_parameters" / "faiss.index"))
        self.embedding_model_name = embedding_model_name or str(PROJECT_ROOT / "data" / "model" / "bge-small-en")
        self.bm25_weight = 1.0
        self.semantic_weight = 1.0
        self.rrf_k = 60

        self.all_chunks = []
        self.bm25 = None
        self.index = None
        self.model = None
        self._initialized = False

    def initialize(self):
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.all_chunks = [json.loads(line) for line in f if line.strip()]

        self.model = SentenceTransformer(self.embedding_model_name)

        bm25_corpus = [tokenize_for_bm25(chunk["bm25_text"]) for chunk in self.all_chunks]
        self.bm25 = BM25Okapi(bm25_corpus)

        self.index = faiss.read_index(str(self.faiss_index_path))

        self._initialized = True

    def _ensure_initialized(self):
        if not self._initialized:
            self.initialize()

    def _encode_query(self, question: str):
        query_instruction = "Given a product/configuration question, retrieve relevant passages that answer the question."
        instructed_query = f"<instruct>{query_instruction}\n<query>{question}"
        return self.model.encode(
            [instructed_query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

    def _bm25_search(self, question, top_k=10):
        query_tokens = tokenize_for_bm25(question)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def _semantic_search(self, question, top_k=10):
        query_embedding = self._encode_query(question)
        scores, indices = self.index.search(query_embedding, top_k)
        return [(int(idx), float(score)) for score, idx in zip(scores[0], indices[0]) if idx != -1]

    def _reciprocal_rank_fusion(self, ranked_lists, weights=None, k=60):
        if weights is None:
            weights = [1.0] * len(ranked_lists)
        rrf_scores = {}
        for ranked_list, weight in zip(ranked_lists, weights):
            for rank, (doc_idx, _) in enumerate(ranked_list, start=1):
                if doc_idx not in rrf_scores:
                    rrf_scores[doc_idx] = 0.0
                rrf_scores[doc_idx] += weight / (k + rank)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, question, top_k=8):
        self._ensure_initialized()
        fetch_k = top_k * 3

        bm25_results = self._bm25_search(question, top_k=fetch_k)
        semantic_results = self._semantic_search(question, top_k=fetch_k)

        bm25_rank_map = {idx: (r, s) for r, (idx, s) in enumerate(bm25_results, 1)}
        semantic_rank_map = {idx: (r, s) for r, (idx, s) in enumerate(semantic_results, 1)}

        fused = self._reciprocal_rank_fusion(
            [bm25_results, semantic_results],
            weights=[self.bm25_weight, self.semantic_weight],
            k=self.rrf_k
        )

        results = []
        for doc_idx, rrf_score in fused[:top_k]:
            item = dict(self.all_chunks[doc_idx])
            item["score"] = rrf_score
            item["bm25_rank"] = bm25_rank_map[doc_idx][0] if doc_idx in bm25_rank_map else None
            item["bm25_score"] = bm25_rank_map[doc_idx][1] if doc_idx in bm25_rank_map else None
            item["semantic_rank"] = semantic_rank_map[doc_idx][0] if doc_idx in semantic_rank_map else None
            item["semantic_score"] = semantic_rank_map[doc_idx][1] if doc_idx in semantic_rank_map else None
            results.append(item)

        return results

    def _hybrid_retrieve(self, question, top_k=8):
        results = self.retrieve(question, top_k=top_k)
        context_blocks = []
        for i, item in enumerate(results, 1):
            text = item.get('embedding_text', '')
            code = item.get('code', 'N/A')
            param = item.get('parameter', 'N/A')
            block = f"[Document {i} - Parameter: {param} (Code: {code})]\n{text}"
            context_blocks.append(block)
        return "\n\n".join(context_blocks)

    def _build_rag_prompt(self, question, context):
        return f"""You are an expert system configuration assistant for Datalogic Magellan barcode scanners.

    Context Information:
    Below are relevant configuration parameters extracted from the Datalogic Functional Requirements Specification (FRS) documentation.
    Each retrieved document represents ONE parameter.
    Each document may include: Parameter Name, Description, Notes, Code, Type, Current Value, Protection, Options, Interface Default Values, Topic, Section, Model, Document ID.

    <context>
    {context}
    </context>

    Answer the question accurately based ONLY on the provided context.
    If the context does not contain the information needed, say "I don't have enough information to answer this."
    When mentioning a configuration code or hex value, format it clearly.

    Question: {question}

    Answer:"""

    def _build_action_context(self, retrieved_chunks):
        blocks = []
        for i, item in enumerate(retrieved_chunks, 1):
            field_name = item.get("field_name", "N/A")
            code = item.get("code", "N/A")
            label = item.get("parameter", "N/A")
            desc = item.get("description", "")
            valid_values = item.get("valid_values", [])
            agent_exec = item.get("agent_execution", {})
            page_path = item.get("page_path", "")

            lines = [
                f"[Parameter {i}]",
                f"  Field Name   : {field_name}",
                f"  Label        : {label}",
                f"  Code         : {code}",
            ]
            if desc:
                lines.append(f"  Description  : {desc}")
            if page_path:
                lines.append(f"  Page Path    : {page_path}")

            if isinstance(valid_values, list) and valid_values:
                lines.append("  Valid Options:")
                for vv in valid_values:
                    lines.append(f'    - "{vv.get("label", "")}": value="{vv.get("value", "")}"')

            if agent_exec:
                lines.append("  Agent Execution Template:")
                lines.append(f'    step1: {agent_exec.get("step1", "")}')
                lines.append(f'    step2: {agent_exec.get("step2", "")}')

            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def query(self, question, top_k=8, model_name="gemini-2.5-flash"):
        import google.generativeai as genai
        self._ensure_initialized()
        context = self._hybrid_retrieve(question, top_k=top_k)
        prompt = self._build_rag_prompt(question, context)

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        llm = genai.GenerativeModel(model_name)
        response = llm.generate_content(prompt)
        return response.text.strip()

    def get_action_context(self, query, top_k=5):
        print(f"RAGPipeline.get_action_context called with query: {query}")
        self._ensure_initialized()
        retrieved_chunks = self.retrieve(query, top_k=top_k)
        print(f"Retrieved {len(retrieved_chunks)} chunks for action context.")
        return self._build_action_context(retrieved_chunks)
