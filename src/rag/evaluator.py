"""
RAG Evaluator Module
====================
Cung cấp các công cụ để tự động sinh tập Test Set từ dữ liệu chunks,
đánh giá chất lượng Retrieval (Hit Rate, MRR) và chất lượng LLM Generation (LLM-as-a-Judge).
"""

import os
import re
import json
import random
from typing import List, Dict, Any, Tuple
import numpy as np
from groq import Groq

# ---------------------------------------------------------------------------
# 1. Tự động sinh Test Set
# ---------------------------------------------------------------------------

def generate_test_set(chunks: List[Dict[str, Any]], num_samples: int = 15, api_key: str = None) -> List[Dict[str, Any]]:
    """
    Duyệt ngẫu nhiên qua các chunks dữ liệu scanner parameter và gọi LLM để sinh ra
    câu hỏi thực tế tương ứng với chunk đó cùng đáp án chuẩn (Ground Truth).
    """
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY", "")
        
    client = Groq(api_key=api_key)
    
    valid_chunks = [c for c in chunks if c.get("parameter_name") and len(c.get("text", "")) > 100]
    if not valid_chunks:
        valid_chunks = chunks
        
    sampled_chunks = random.sample(valid_chunks, min(num_samples, len(valid_chunks)))
    test_set = []
    
    print(f"Đang sinh {len(sampled_chunks)} câu hỏi kiểm thử từ dữ liệu gốc...")
    
    for i, chunk in enumerate(sampled_chunks, start=1):
        prompt = f"""
You are an expert in configuring Datalogic  barcode readers. There are some documentations of some parameter as below:

---
{chunk['text']}
---

Create a test sample include:
1. A question about configuration of this parameter in Vietnamese (e.g., "How to enable/disable..." or "What are the options for..."). The question should be natural, and can directly ask about the parameter name or its function.
2. A short, precise answer as the Ground Truth based on the documentation above.

Return a JSON object as follows:
{{
  "question": "câu hỏi tiếng Anh ở đây",
  "ground_truth": "đáp án chuẩn ở đây"
}}
"""
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You only output valid JSON. Do not include markdown code block syntax."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            res_content = response.choices[0].message.content.strip()
            data = json.loads(res_content)
            
            test_set.append({
                "id": i,
                "question": data["question"],
                "expected_doc_id": chunk.get("chunk_id", chunk.get("doc_id")),
                "expected_parameter": chunk.get("parameter_name", ""),
                "ground_truth": data["ground_truth"],
                "source_text": chunk['text']
            })
            print(f"  [OK] Đã sinh câu hỏi {i}/{len(sampled_chunks)} cho parameter: {chunk.get('parameter_name')}")
        except Exception as e:
            print(f"  [Error] Lỗi khi sinh câu hỏi ở chunk {chunk.get('parameter_name')}: {e}")
            
    return test_set

# ---------------------------------------------------------------------------
# 2. Đánh giá chất lượng Retrieval (BM25 vs Semantic vs Hybrid)
# ---------------------------------------------------------------------------

def evaluate_retrieval(retrievers_dict: Dict[str, Any], test_set: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Dict[str, float]]:
    """
    Tính toán Hit Rate@K và MRR@K cho các bộ lọc tìm kiếm khác nhau.
    
    Args:
        retrievers_dict: Dict dạng {"Name": search_function}
                        trong đó search_function(query, top_k) -> trả về list các dict chunk
        test_set: Tập test đã sinh ở bước 1
    """
    results = {}
    
    for name, search_func in retrievers_dict.items():
        hits = 0
        reciprocal_ranks = []
        
        for sample in test_set:
            expected_id = sample["expected_doc_id"]
            expected_param = sample["expected_parameter"]
            query = sample["question"]
            
            # Tìm kiếm
            retrieved_chunks = search_func(query, top_k=top_k)
            
            # Kiểm tra xem expected_id hoặc expected_param có nằm trong kết quả tìm được không
            found_rank = None
            for rank, chunk in enumerate(retrieved_chunks, start=1):
                chunk_id = chunk.get("chunk_id", chunk.get("doc_id"))
                param_name = chunk.get("parameter_name", "")
                
                if chunk_id == expected_id or (expected_param and param_name == expected_param):
                    found_rank = rank
                    break
            
            if found_rank is not None:
                hits += 1
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)
                
        hit_rate = (hits / len(test_set)) * 100
        mrr = np.mean(reciprocal_ranks) * 100
        
        results[name] = {
            f"Hit Rate@{top_k} (%)": round(hit_rate, 2),
            f"MRR@{top_k} (%)": round(mrr, 2)
        }
        
    return results

# ---------------------------------------------------------------------------
# 3. Đánh giá chất lượng sinh câu trả lời (Generation Quality)
# ---------------------------------------------------------------------------

def evaluate_generation(
    rag_pipeline_func, 
    test_set: List[Dict[str, Any]], 
    api_key: str = None
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Gọi RAG Pipeline để sinh câu trả lời, sau đó dùng LLM chấm điểm từ 1-5
    cho mỗi câu dựa trên độ chính xác thông tin so với Ground Truth.
    
    rag_pipeline_func: function(question: str) -> str (trả về câu trả lời cuối cùng)
    """
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY", "")
        
    client = Groq(api_key=api_key)
    evaluated_samples = []
    scores = []
    
    print("\nBắt đầu chạy thử RAG và đánh giá câu trả lời...")
    for sample in test_set:
        query = sample["question"]
        ground_truth = sample["ground_truth"]
        
        # Chạy RAG
        try:
            generated_answer = rag_pipeline_func(query)
        except Exception as e:
            generated_answer = f"Error running RAG: {e}"
            
        # LLM-as-a-Judge chấm điểm từ 1 đến 5
        prompt = f"""
Bạn là giám khảo kiểm thử hệ thống RAG độc lập. Hãy chấm điểm câu trả lời của AI dựa vào câu hỏi và Đáp án chuẩn (Ground Truth).

[Câu hỏi]: {query}
[Đáp án chuẩn (Ground Truth)]: {ground_truth}
[Câu trả lời của AI]: {generated_answer}

Quy tắc chấm điểm (từ 1 đến 5 điểm):
- 5 điểm: Câu trả lời hoàn hảo, chính xác tuyệt đối thông số kỹ thuật, đầy đủ thông tin so với Đáp án chuẩn.
- 4 điểm: Câu trả lời đúng phần lớn, nhưng thiếu một chút chi tiết nhỏ không quá quan trọng.
- 3 điểm: Câu trả lời có ý đúng nhưng chưa đầy đủ hoặc diễn đạt mơ hồ.
- 2 điểm: Câu trả lời sai thông số chính hoặc thiếu thông tin trầm trọng.
- 1 điểm: Câu trả lời hoàn toàn sai, lạc đề hoặc bịa đặt thông tin.

Hãy trả về định dạng JSON duy nhất như sau:
{{
  "score": integer (từ 1 đến 5),
  "reason": "Giải thích lý do ngắn gọn bằng tiếng Việt"
}}
"""
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You only output valid JSON. Do not include markdown code block syntax."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            res_content = response.choices[0].message.content.strip()
            eval_data = json.loads(res_content)
            score = int(eval_data["score"])
            reason = eval_data["reason"]
        except Exception as e:
            print(f"  [Error Judge] Lỗi khi chấm điểm: {e}")
            score = 1
            reason = f"Lỗi chấm điểm tự động: {e}"
            
        scores.append(score)
        evaluated_samples.append({
            "question": query,
            "ground_truth": ground_truth,
            "ai_answer": generated_answer,
            "score": score,
            "reason": reason
        })
        print(f"  - Q: {query[:40]}... -> Điểm: {score}/5")
        
    # Tính điểm % trung bình
    accuracy_pct = (sum(scores) / (len(test_set) * 5)) * 100
    
    summary = {
        "Average Score": round(sum(scores) / len(test_set), 2),
        "Accuracy Rate (%)": round(accuracy_pct, 2),
        "Perfect Answers (5/5)": scores.count(5),
        "Good Answers (>=4/5)": sum(1 for s in scores if s >= 4),
        "Total Questions": len(test_set)
    }
    
    return evaluated_samples, summary
