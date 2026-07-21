def build_document(record):
    # Build embedding_text
    lines = []
    param = record.get("parameter", "")
    if param:
        lines.append(f"Parameter Name:\n{param}")

    desc = record.get("description", "")
    notes = record.get("notes", "")
    if desc or notes:
        desc_block = "Description:\n"
        if desc:
            desc_block += desc
        if notes:
            if desc:
                desc_block += "\n"
            desc_block += f"Additional Notes:\n{notes}"
        lines.append(desc_block)

    code = record.get("code", "")
    if code:
        lines.append(f"Parameter Code:\n{code}")

    p_type = record.get("type", "")
    if p_type:
        lines.append(f"Type:\n{p_type}")

    prot = record.get("protection", "")
    if prot:
        lines.append(f"Protection:\n{prot}")

    val = record.get("value", "")
    if val:
        lines.append(f"Current Value:\n{val}")

    options = record.get("options")
    if options and isinstance(options, dict):
        opt_lines = ["Available Options"]
        for k, v in options.items():
            opt_lines.append(f"{v} -> {k}")
        lines.append("\n".join(opt_lines))

    defaults = record.get("interface_defaults")
    if defaults and isinstance(defaults, dict):
        def_lines = ["Interface Default Values"]
        for k, v in defaults.items():
            def_lines.append(f"{k} -> {v}")
        lines.append("\n".join(def_lines))

    topic = record.get("head") or record.get("topic", "")
    if topic:
        lines.append(f"Topic:\n{topic}")

    section = record.get("section", "")
    if section:
        lines.append(f"Section:\n{section}")

    model = record.get("model", "")
    if model:
        lines.append(f"Model:\n{model}")

    doc_id = record.get("doc_id", "")
    if doc_id:
        lines.append(f"Document:\n{doc_id}")

    embedding_text = "\n\n".join(lines)

    # Build bm25_text
    bm25_lines = []
    if param:
        bm25_lines.append(f"Parameter Name: {param} {param} {param}")
    if desc:
        bm25_lines.append(f"Description: {desc}")
    if notes:
        bm25_lines.append(f"Notes: {notes}")
    if code:
        bm25_lines.append(f"Code: {code}")
    if options and isinstance(options, dict):
        opts_str = " ".join([f"{v} {k}" for k, v in options.items()])
        bm25_lines.append(f"Options: {opts_str}")
    if defaults and isinstance(defaults, dict):
        defs_str = " ".join([f"{k} {v}" for k, v in defaults.items()])
        bm25_lines.append(f"Interface Defaults: {defs_str}")
    if topic:
        bm25_lines.append(f"Topic: {topic}")
    if section:
        bm25_lines.append(f"Section: {section}")
    if model:
        bm25_lines.append(f"Model: {model}")
    if doc_id:
        bm25_lines.append(f"Document: {doc_id}")
    bm25_text = "\n\n".join(bm25_lines)

    # Build metadata
    metadata = {
        "parameter": param,
        "parameter_lower": param.lower() if param else "",
        "code": code,
        "model": model,
        "topic": topic,
        "section": section,
        "doc_id": doc_id,
        "type": p_type,
        "source": record.get("source_file", "")
    }

    return embedding_text, bm25_text, metadata

"""## Define file JSON path

"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

from pathlib import Path
import json
import re

JSON_PATH = r"/content/drive/MyDrive/DLVN/config_Magellan-900i_DR9401800_ReadingParameters.json"

OUTPUT_PATH = Path("/content/drive/MyDrive/DLVN/chunks.jsonl")

json_path = Path(JSON_PATH)
print("JSON exists:", json_path.exists())
print("JSON path:", json_path)

"""## Load JSON

"""

with json_path.open("r", encoding="utf-8") as f:
    raw_data = json.load(f)

type(raw_data), raw_data if isinstance(raw_data, (str, int, float)) else "loaded"

"""## Convert JSON to documents"""

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def flatten_config(config_data):
    records = []
    for panel_key, panel_value in config_data.items():
        if not isinstance(panel_value, dict):
            continue
        for param_key, param_value in panel_value.items():
            if not isinstance(param_value, dict) or param_key == "frs_panel_context":
                continue
            record = {}
            record["code"] = param_value.get("code", "")
            record["type"] = param_value.get("type", "")
            record["protection"] = param_value.get("protection", "")
            record["value"] = param_value.get("value", "")
            record["options"] = param_value.get("options", {})
            record["interface_defaults"] = param_value.get("interfaceDefaults", {})
            frs = param_value.get("frs_context", {})
            record["parameter"] = frs.get("parameter") or param_value.get("context", "")
            record["description"] = frs.get("description", "")
            record["notes"] = frs.get("notes", "")
            record["topic"] = frs.get("topic", "")
            record["head"] = frs.get("head", "")
            record["section"] = frs.get("section", "")
            record["model"] = frs.get("model", "")
            record["doc_id"] = frs.get("doc_id", "")
            record["source_file"] = frs.get("source_file", "")
            records.append(record)
    return records

if isinstance(raw_data, dict):
    records_to_process = flatten_config(raw_data)
else:
    records_to_process = raw_data

# Build documents (1 record = 1 document)
all_chunks = []
for i, record in enumerate(records_to_process):
    embedding_text, bm25_text, metadata = build_document(record)
    embedding_text = normalize_text(embedding_text)
    bm25_text = normalize_text(bm25_text)

    chunk = {
        "chunk_id": str(i),
        "text": embedding_text,
        "embedding_text": embedding_text,
        "bm25_text": bm25_text
    }
    chunk.update(metadata)

    if "parameter" in metadata:
        chunk["parameter_name"] = metadata["parameter"]

    all_chunks.append(chunk)

print(f"Created {len(all_chunks)} documents from {len(records_to_process)} records.")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for chunk in all_chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print(f"saved {len(all_chunks)} chunks to {OUTPUT_PATH.resolve()}")

"""## Install library for embedding and FAISS


"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q sentence-transformers faiss-cpu rank-bm25

"""## Create embeddings using BAAI/bge-m3 model

"""

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
QUERY_INSTRUCTION = "Given a product/configuration question, retrieve relevant passages that answer the question."

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

texts = [chunk["embedding_text"] for chunk in all_chunks]

def encode_documents(texts):
    return model.encode(
        texts,
        batch_size=1,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")


def encode_query(question: str):
    instructed_query = f"<instruct>{QUERY_INSTRUCTION}\n<query>{question}"
    return model.encode(
        [instructed_query],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")


embeddings = encode_documents(texts)
print("embedding shape:", embeddings.shape)

"""## Build FAISS index

`IndexFlatIP` search by cosine similarity.
"""

!pip install faiss-cpu
import faiss

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("vectors in index:", index.ntotal)

"""## Build BM25 Index


"""

import re
from rank_bm25 import BM25Okapi

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


bm25_corpus = [tokenize_for_bm25(chunk["bm25_text"]) for chunk in all_chunks]
bm25 = BM25Okapi(bm25_corpus)
print("BM25 index built:", len(bm25_corpus), "documents")

"""## Hybrid Retrieval via Reciprocal Rank Fusion (RRF)


"""

def bm25_search(question, top_k=10):
    """Search using BM25 """
    query_tokens = tokenize_for_bm25(question)
    if not query_tokens:
        return []
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


def semantic_search(question, top_k=10):
    """Search using FAISS semantic similarity """
    query_embedding = encode_query(question)
    scores, indices = index.search(query_embedding, top_k)
    return [(int(idx), float(score)) for score, idx in zip(scores[0], indices[0]) if idx != -1]


def reciprocal_rank_fusion(ranked_lists, weights=None, k=60):
    """Reciprocal Rank Fusion: RRF_score(d) = Σ weight_i / (k + rank_i(d))"""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    rrf_scores = {}
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, (doc_idx, _) in enumerate(ranked_list, start=1):
            if doc_idx not in rrf_scores:
                rrf_scores[doc_idx] = 0.0
            rrf_scores[doc_idx] += weight / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


BM25_WEIGHT = 1.0
SEMANTIC_WEIGHT = 1.0
RRF_K = 60

def hybrid_retrieve(question, top_k=8, bm25_weight=BM25_WEIGHT, semantic_weight=SEMANTIC_WEIGHT):
    """Hybrid retrieval: BM25 + Semantic search, fused via RRF."""
    fetch_k = top_k * 3

    bm25_results = bm25_search(question, top_k=fetch_k)
    semantic_results = semantic_search(question, top_k=fetch_k)

    bm25_rank_map = {idx: (r, s) for r, (idx, s) in enumerate(bm25_results, 1)}
    semantic_rank_map = {idx: (r, s) for r, (idx, s) in enumerate(semantic_results, 1)}

    fused = reciprocal_rank_fusion(
        [bm25_results, semantic_results],
        weights=[bm25_weight, semantic_weight],
        k=RRF_K
    )

    results = []
    for doc_idx, rrf_score in fused[:top_k]:
        item = dict(all_chunks[doc_idx])
        item["score"] = rrf_score
        item["bm25_rank"] = bm25_rank_map[doc_idx][0] if doc_idx in bm25_rank_map else None
        item["bm25_score"] = bm25_rank_map[doc_idx][1] if doc_idx in bm25_rank_map else None
        item["semantic_rank"] = semantic_rank_map[doc_idx][0] if doc_idx in semantic_rank_map else None
        item["semantic_score"] = semantic_rank_map[doc_idx][1] if doc_idx in semantic_rank_map else None
        results.append(item)

    return results


print(" Hybrid retrieval functions ready!")
print(f"   BM25_WEIGHT={BM25_WEIGHT}, SEMANTIC_WEIGHT={SEMANTIC_WEIGHT}, RRF_K={RRF_K}")

"""## Test Hybrid Retrieve - So sánh kết quả

**Hybrid** vs **Semantic-only** vs **BM25-only**

"""

# Hybrid Retrieval (BM25 + Semantic via RRF)
def retrieve(question: str, top_k=8):
    """Hybrid retrieve: combines BM25 keyword matching + Semantic search via RRF."""
    return hybrid_retrieve(question, top_k=top_k)


# Hybrid vs Semantic-only vs BM25-only
test_queries = [
    "Good Read Beep Volume",           # Exact parameter name
    "CI_GOOD_READ_BEEP_VOLUME",        # Internal code
    "how to control sound volume?",    # Natural language
]

for question in test_queries:
    print("=" * 90)
    print(f'QUERY: "{question}"')
    print("-" * 90)

    hybrid_results = hybrid_retrieve(question, top_k=5)
    sem_results = semantic_search(question, top_k=5)
    bm25_results_raw = bm25_search(question, top_k=5)

    print("\n  HYBRID (BM25 + Semantic via RRF):")
    for i, r in enumerate(hybrid_results, 1):
        print(f"    {i}. {r.get('parameter', r.get('parameter_name', ''))} | bm25_rank={r.get('bm25_rank','-')} sem_rank={r.get('semantic_rank','-')} | score={r['score']:.4f}")

    print("\n  SEMANTIC ONLY:")
    for i, (idx, score) in enumerate(sem_results, 1):
        print(f"    {i}. {all_chunks[idx].get('parameter', all_chunks[idx].get('parameter_name', ''))} | score={score:.4f}")

    print("\n  BM25 ONLY:")
    for i, (idx, score) in enumerate(bm25_results_raw, 1):
        print(f"    {i}. {all_chunks[idx].get('parameter', all_chunks[idx].get('parameter_name', ''))} | score={score:.4f}")

    print()

"""## Save FAISS index and metadata"""

INDEX_PATH = Path("/content/drive/MyDrive/DLVN/faiss.index")
METADATA_PATH = Path("/content/drive/MyDrive/DLVN/faiss_metadata.json")

faiss.write_index(index, str(INDEX_PATH))

with METADATA_PATH.open("w", encoding="utf-8") as f:
    json.dump(
        {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "query_instruction": QUERY_INSTRUCTION,
            "chunks": all_chunks,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("saved index:", INDEX_PATH.resolve())
print("saved metadata:", METADATA_PATH.resolve())

def build_context(results):
    context_blocks = []
    for i, item in enumerate(results, 1):
        text = item.get('embedding_text', '')
        code = item.get('code', 'N/A')
        param = item.get('parameter', 'N/A')

        block = f"[Document {i} - Parameter: {param} (Code: {code})]\n{text}"
        context_blocks.append(block)
    return "\n\n".join(context_blocks)

def build_rag_prompt(question, context):
    return f"""You are an expert system configuration assistant for Datalogic Magellan barcode scanners.

    Context Information:
    Below are relevant configuration parameters extracted from the Datalogic Functional Requirements Specification (FRS) documentation.
    Each retrieved document represents ONE parameter.
    Each document may include: Parameter Name, Description, Notes, Code, Type, Current Value, Protection, Options, Interface Default Values, Topic, Section, Model, Document ID.

    <context>
    {context}
    </context>

    Answer the question accurately based ONLY on the provided context.
    If the context does not contain the information needed, say \"I don't have enough information to answer this.\"
    When mentioning a configuration code or hex value, format it clearly.

    Question: {question}

    Answer:"""

"""## Call LLM to generate the answer

"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q openai
!pip install -q groq

import os
from groq import Groq

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

def answer_with_llama(question: str, top_k=8, model_name="meta-llama/llama-4-scout-17b-16e-instruct"):
    client = Groq()

    try:
        retrieved_chunks = retrieve(question, top_k=top_k)
        context = "\n---\n".join([item["text"] for item in retrieved_chunks])
    except NameError:
        context = "Can not founf retrieve, check agian"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": """You are an expert assistant for Datalogic Magellan barcode scanner configuration.

The retrieved context comes from official Datalogic Functional Requirements Specification (FRS) documents.

Each retrieved document describes ONE scanner configuration parameter.

The most important field is:

• Parameter Name
  (This is the human-readable name shown in the documentation, for example:
   "Good Read Beep Enable",
   "Good Read Beep Frequency",
   "Scanner Button Options")

Other fields may include:

• Panel
• Parameter Code
• Type
• Current Value
• Interface Default Values
• Valid Options
• Numeric Range
• Increment
• Protection
• Document ID
• Model

Your task is to answer the user's question ONLY using the retrieved context.

Rules

1. Always identify the parameter using its Parameter Name.

2. Ignore any internal JSON keys (such as CI_GOOD_READ_BEEP_CONTROL).
They are implementation identifiers and should never appear in your answer unless the user explicitly asks for them.

3. Treat every parameter as an independent entity.

4. Never combine information from different parameters.

5. If multiple parameters are retrieved, answer ONLY for the parameter that best matches the user's question.

6. If the parameter contains an Options section, list every option exactly as documented.

7. Distinguish clearly between:
   • Current Value
   • Interface Default Values
   • Available Options
   • Numeric Range

8. Never assume undocumented values or meanings.

9. If the requested information is not present in the retrieved context, answer exactly:

The provided documentation does not contain enough information to answer this question.

10. When answering "What is..." or "How to..." questions, always provide a natural explanation using the 'Description' and 'Additional Notes' fields first.
Then, if applicable, structure the technical details like this:

Description: [Your natural explanation here]
Parameter Name:
Code:
Type:
Current Value:

Options:
- ...

11. Keep the answer concise, factual, and technical."""}
            ,
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0.3,
        max_completion_tokens=1024,
        top_p=1,
        stream=False
    )

    return response.choices[0].message.content

def ask_rag(question: str, top_k=8):
    answer = answer_with_llama(question, top_k=top_k)

    print("Answer:")
    print(answer)

#ask_rag("What options does Good Read Beep volume have ?")
ask_rag("Explain the option On only when in ScannerActiveMode/HandheldState in the parameter called Center Zone??")
#ask_rag("How do I configure the beeper")

"""# ## RAG Evaluation

1. **Test Set Generation** (LLM-generated 15 real questions + Ground Truth from scanner parameters.)
2. **Retrieval**: Compares Hit Rate and MRR of BM25, Semantic, and Hybrid (RRF)
3. **Generation**:LLM-scored answers (1–5) based on accuracy.

"""

import os
import sys
import shutil
import evaluator
import pandas as pd

possible_paths = [
    '/content/drive/MyDrive/DLVN',
    '/content/drive/MyDrive',
    '/content',
    os.getcwd()
]

found_evaluator = False
for p in possible_paths:
    eval_file = os.path.join(p, 'evaluator.py')
    if os.path.exists(eval_file):

        if p != '/content' and os.path.exists('/content'):
            shutil.copy(eval_file, '/content/evaluator.py')
            sys.path.insert(0, '/content')
        else:
            sys.path.insert(0, p)
        found_evaluator = True
        print(f"Found and configured evaluator.py at: {eval_file}")
        break

if not found_evaluator:

    print("Can not found file evaluator.py in Drive. Upload file from computer:")
    try:
        from google.colab import files
        uploaded = files.upload()
        if 'evaluator.py' in uploaded:
            sys.path.insert(0, '/content')
            found_evaluator = True
            print("Upload evaluator.py successfully!")
    except Exception as e:
        print("Cannot open upload dialog or not running on Colab.:", e)


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

test_set = evaluator.generate_test_set(all_chunks, num_samples=15, api_key=GROQ_API_KEY)

print("\n--- Example of 3 test questions generated from the source data: ---")
for t in test_set[:3]:
    print(f"Q: {t['question']}")
    print(f"Expected Parameter: {t['expected_parameter']}")
    print(f"Ground Truth: {t['ground_truth']}\n")

retrievers = {
    "BM25 Only": lambda q, top_k: [
        {"chunk_id": idx, "parameter_name": all_chunks[idx].get("parameter")}
        for idx, _ in bm25_search(q, top_k=top_k)
    ],
    "Semantic Only": lambda q, top_k: [
        {"chunk_id": idx, "parameter_name": all_chunks[idx].get("parameter")}
        for idx, _ in semantic_search(q, top_k=top_k)
    ],
    "Hybrid (BM25 + Semantic)": lambda q, top_k: hybrid_retrieve(q, top_k=top_k)
}

retrieval_results = evaluator.evaluate_retrieval(retrievers, test_set, top_k=5)

df_retrieval = pd.DataFrame(retrieval_results).T
print("=== RETRIEVAL QUALITY REPORT ===")
print(df_retrieval.to_string())

def run_rag_pipeline(question):
    return answer_with_llama(question, top_k=5)

evaluated_samples, gen_summary = evaluator.evaluate_generation(
    run_rag_pipeline,
    test_set,
    api_key=GROQ_API_KEY
)

print("\n=== ANSWER LLM GENERATION QUALITY REPORT ===")
print(f"- Average: {gen_summary['Average Score']}/5")
print(f"- Accuracy Rate: {gen_summary['Accuracy Rate (%)']}%")
print(f"- Perfect Answers (5/5): {gen_summary['Perfect Answers (5/5)']}/{gen_summary['Total Questions']}")
print(f"- Good Answers (>=4/5): {gen_summary['Good Answers (>=4/5)']}/{gen_summary['Total Questions']}")