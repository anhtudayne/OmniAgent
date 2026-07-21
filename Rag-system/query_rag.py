import os
import sys
import argparse
import json
import math
import time

# Khai báo các thư viện cần thiết.
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    import google.generativeai as genai
    from dotenv import load_dotenv
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Lỗi: Vui lòng chạy lệnh: pip install chromadb sentence-transformers google-generativeai python-dotenv rank_bm25")
    sys.exit(1)

# Tải biến môi trường
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Lỗi: Không tìm thấy GEMINI_API_KEY trong file .env hoặc biến môi trường.")
    sys.exit(1)
genai.configure(api_key=api_key)


def parse_query_with_llm(user_query):
    """
    Sử dụng LLM để định tuyến câu hỏi (Router).
    Bóc tách ra Search Query (ngữ nghĩa), Model Filter và Keywords.
    """
    print("1. Đang dùng LLM Router để phân tích câu hỏi...")
    
    prompt = f"""Bạn là một chuyên gia phân tích câu hỏi (Query Router) cho hệ thống tìm kiếm tài liệu cấu hình máy quét mã vạch.
Dựa vào câu hỏi của người dùng, hãy trích xuất 3 thông tin sau:
1. "search_query": Viết lại câu hỏi thành cụm từ tìm kiếm tiếng Anh ngắn gọn. CHÚ Ý QUAN TRỌNG: Nếu phát hiện người dùng gõ dính chữ (ví dụ: "Dataformat"), BẮT BUỘC phải tách ra thành "Data Format".
2. "model_filter": Trích xuất tên dòng máy quét. CHÚ Ý QUAN TRỌNG: Bắt buộc phải viết liền bằng dấu gạch ngang (Ví dụ: "Magellan-1500i", "Magellan-900i", tuyệt đối không viết khoảng trắng "Magellan 900i"). Nếu người dùng không nhắc, để null.
3. "keywords": Mảng tối đa 4 từ khóa kỹ thuật rời rạc. Nếu từ khóa bị dính (Dataformat), phải tách thành các từ khóa rời: ["Data", "Format"].

CÂU HỎI CỦA NGƯỜI DÙNG: "{user_query}"

BẮT BUỘC trả về ĐÚNG định dạng JSON thuần túy (không dùng markdown block ```json), gồm 3 key: search_query, model_filter, keywords.
"""
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    
    try:
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        parsed_data = json.loads(raw_text)
        return parsed_data
    except Exception as e:
        print(f"Lỗi khi Parse JSON từ LLM: {e}. Sẽ dùng câu hỏi gốc làm mặc định.")
        return {
            "search_query": user_query,
            "model_filter": None,
            "keywords": user_query.split()[:3]
        }

# Cache toàn cục (Global) để không phải rebuild BM25 và reload Model mỗi lần query
_GLOBAL_BM25 = None
_GLOBAL_IDS = None
_GLOBAL_METADATAS = None
_GLOBAL_EMBEDDING_MODEL = None
_GLOBAL_COLLECTION = None

def get_models_and_collection(persist_dir, collection_name, embedding_model_path):
    global _GLOBAL_EMBEDDING_MODEL, _GLOBAL_COLLECTION
    
    if _GLOBAL_COLLECTION is None:
        print("   [ChromaDB] Đang khởi tạo kết nối Database (chỉ 1 lần)...")
        client = chromadb.PersistentClient(path=persist_dir)
        _GLOBAL_COLLECTION = client.get_collection(name=collection_name)
        
    if _GLOBAL_EMBEDDING_MODEL is None:
        print("   [SentenceTransformer] Đang tải mô hình nhúng (Embedding Model) vào RAM (chỉ 1 lần)...")
        _GLOBAL_EMBEDDING_MODEL = SentenceTransformer(embedding_model_path)
        
    return _GLOBAL_EMBEDDING_MODEL, _GLOBAL_COLLECTION

def get_bm25_index(collection):
    global _GLOBAL_BM25, _GLOBAL_IDS, _GLOBAL_METADATAS
    if _GLOBAL_BM25 is None:
        print("   [BM25 Cache] Đang tải toàn bộ dữ liệu lên RAM và xây dựng BM25 Index (chỉ chạy 1 lần)...")
        all_docs = collection.get()
        _GLOBAL_IDS = all_docs["ids"]
        _GLOBAL_METADATAS = all_docs.get("metadatas", [])
        corpus_texts = all_docs["documents"]
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        _GLOBAL_BM25 = BM25Okapi(tokenized_corpus)
    return _GLOBAL_BM25, _GLOBAL_IDS, _GLOBAL_METADATAS

def preload_models(persist_dir, collection_name, embedding_model_path):
    """
    Gọi hàm này một lần lúc khởi động ứng dụng để nạp sẵn mọi thứ vào RAM, 
    giúp cho các câu hỏi sau không bị tính thời gian Cold Start.
    """
    print("\n[HỆ THỐNG] Đang tải trước (Pre-load) toàn bộ Models và BM25 Index vào RAM...")
    _, collection = get_models_and_collection(persist_dir, collection_name, embedding_model_path)
    get_bm25_index(collection)
    print("[HỆ THỐNG] Quá trình tải trước hoàn tất!\n")


def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """
    Kết hợp kết quả từ 2 thuật toán (Dense + Sparse) bằng công thức RRF.
    """
    scores = {}
    
    # Xử lý Dense
    for rank, doc_id in enumerate(dense_results):
        if doc_id not in scores:
            scores[doc_id] = 0.0
        scores[doc_id] += 1.0 / (k + rank + 1)
        
    # Xử lý Sparse (BM25)
    for rank, doc_id in enumerate(sparse_results):
        if doc_id not in scores:
            scores[doc_id] = 0.0
        scores[doc_id] += 1.0 / (k + rank + 1)
        
    # Sắp xếp lại theo điểm RRF giảm dần
    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs]


def hybrid_search(parsed_query, persist_dir, collection_name, embedding_model_path, top_k=3):
    """
    Thực hiện Hybrid Search: Kết hợp Vector Semantic Search và BM25 Keyword Search.
    """
    print(f"2. Đang kết nối tới Chroma DB tại '{persist_dir}'...")
    
    search_query = parsed_query.get("search_query", "")
    model_filter = parsed_query.get("model_filter", None)
    keywords = parsed_query.get("keywords", [])
    
    # Sử dụng model và collection đã được cache (Load 1 lần duy nhất)
    embedding_model, collection = get_models_and_collection(persist_dir, collection_name, embedding_model_path)
    
    print(f"   -> [LLM Router] Search Query (Dense): '{search_query}'")
    print(f"   -> [LLM Router] Model Filter (Metadata): {model_filter}")
    print(f"   -> [LLM Router] Keywords (Sparse BM25): {keywords}")
    
    # ---------------------------------------------------------
    # LUỒNG 1: DENSE SEARCH (Tìm theo Ngữ nghĩa bằng Vector)
    # ---------------------------------------------------------
    print("3a. Đang chạy luồng Dense Vector Search...")
    query_embedding = embedding_model.encode([search_query], normalize_embeddings=True).tolist()
    
    query_kwargs = {
        "query_embeddings": query_embedding,
        "n_results": top_k * 2  # Lấy dư ra một chút để gộp RRF
    }
    if model_filter:
        query_kwargs["where"] = {"model": model_filter}
        
    dense_res = collection.query(**query_kwargs)
    dense_ids = dense_res["ids"][0] if dense_res["ids"] else []
    
    # ---------------------------------------------------------
    # LUỒNG 2: SPARSE SEARCH (Tìm chính xác Từ khóa bằng BM25)
    # ---------------------------------------------------------
    print("3b. Đang chạy luồng Sparse Keyword Search (BM25)...")
    get_kwargs = {}
    if model_filter:
        get_kwargs["where"] = {"model": model_filter}
        
    bm25_ids = []
    
    if len(keywords) > 0:
        bm25, corpus_ids, corpus_metas = get_bm25_index(collection)
        
        # Tách các cụm từ thành các từ đơn để BM25 có thể khớp đúng
        tokenized_query = []
        for kw in keywords:
            tokenized_query.extend(kw.lower().split())
            
        doc_scores = bm25.get_scores(tokenized_query)
        
        # Lọc thủ công dựa trên model_filter
        filtered_indices_and_scores = []
        for i in range(len(doc_scores)):
            if doc_scores[i] == 0:
                continue
                
            match = True
            if model_filter:
                meta_val = corpus_metas[i].get("model") if corpus_metas else None
                if meta_val != model_filter:
                    match = False
                    
            if match:
                filtered_indices_and_scores.append((i, doc_scores[i]))
                
        # Lấy các vị trí có điểm cao nhất
        top_n = top_k * 2
        filtered_indices_and_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Chỉ lấy ID của những tài liệu có chứa ít nhất 1 từ khóa (score > 0)
        bm25_ids = [corpus_ids[i] for i, score in filtered_indices_and_scores[:top_n]]
    sorted_all_ids = reciprocal_rank_fusion(dense_ids, bm25_ids)
    
    if not sorted_all_ids:
        return {}
        
    # Kéo nội dung chi tiết của tất cả các ID thắng cuộc lên
    raw_results = collection.get(ids=sorted_all_ids)
    
    # Bug fix #2: ChromaDB get() không giữ đúng thứ tự ids truyền vào, nên phải reorder
    id_to_index = {doc_id: i for i, doc_id in enumerate(raw_results["ids"])}
    
    # Bug fix #1: Dedup theo ci_key để tránh 1 tham số chiếm hết top slots
    seen_ci_keys = set()
    final_docs = []
    final_metas = []
    
    for doc_id in sorted_all_ids:
        if doc_id not in id_to_index:
            continue
            
        idx = id_to_index[doc_id]
        meta = raw_results["metadatas"][idx]
        doc = raw_results["documents"][idx]
        
        ci_key = meta.get("ci_key")
        if ci_key:
            if ci_key in seen_ci_keys:
                continue
            seen_ci_keys.add(ci_key)
            
        final_docs.append(doc)
        final_metas.append(meta)
        
        if len(final_docs) >= top_k:
            break
            
    return {
        "documents": [final_docs],
        "metadatas": [final_metas]
    }


def ask_gemini(user_query, search_results):
    """
    Hàm LLM cuối cùng: Tổng hợp câu trả lời từ tài liệu lai (Hybrid Results).
    """
    print("4. Đang gửi thông tin cho Gemini AI tổng hợp câu trả lời cuối cùng...")
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    context_blocks = []
    documents = search_results.get('documents', [[]])[0]
    metadatas = search_results.get('metadatas', [[]])[0]
    
    if len(documents) == 0:
        return "Xin lỗi, không tìm thấy tài liệu kỹ thuật nào phù hợp."
    
    for i in range(len(documents)):
        doc_text = documents[i]
        meta = metadatas[i]
        
        raw_json_str = meta.get('raw_json', '{}')
        parsed_json_str = ""
        try:
            raw_dict = json.loads(raw_json_str)
            code = raw_dict.get('code', 'Không xác định')
            opts = raw_dict.get('options', {})
            val = raw_dict.get('value', 'Không xác định')
            parsed_json_str = f"Code: {code}\nGiá trị mặc định (Value): {val}\nCác lựa chọn (Options): {opts}"
        except:
            parsed_json_str = raw_json_str
            
        parameter_id = meta.get('parameter_id', 'Unknown')
        
        block = f"--- TÀI LIỆU {i+1} ---\nNội dung:\n{doc_text}\nParameter ID: {parameter_id}\nChi tiết kỹ thuật:\n{parsed_json_str}\n"
        context_blocks.append(block)
        
    full_context = "\n".join(context_blocks)
    # print(full_context)
    
    prompt = f"""Bạn là một chuyên gia hỗ trợ kỹ thuật xuất sắc cho máy quét mã vạch Datalogic (Magellan).
Dựa vào các TÀI LIỆU CUNG CẤP bên dưới, hãy trả lời câu hỏi của người dùng.

GIẢI THÍCH Ý NGHĨA CÁC TRƯỜNG DỮ LIỆU (ĐỂ HIỂU ĐÚNG TÀI LIỆU):
- Config Key: Tên định danh nội bộ bắt đầu bằng CI_ (dùng trong code).
- Parameter: Tên hiển thị thân thiện của tham số cấu hình.
- Panel: Cây thư mục (gia phả) chứa tham số này trên phần mềm. Mũi tên (>) thể hiện cha > con.
- Section: Phân nhóm nhỏ hơn bên trong Panel để phân loại tham số.
- Parameter ID: Mã định danh ID duy nhất của tham số này (thường nối tên Panel và Parameter lại với nhau).
- Code: Mã lệnh Hexadecimal (VD: 04F4) dùng để lập trình hoặc gửi lệnh xuống máy quét.
- Value / Options: Cặp giá trị cài đặt tương ứng với các lựa chọn của người dùng.

Quy tắc sinh câu trả lời:
1. Nếu người dùng hỏi về một NHÓM tính năng (Ví dụ hỏi về "Data Format"): Hãy kiểm tra xem nó có phải là tên của "Panel" hoặc "Section" không. Mũi tên (>) 
trong Panel thể hiện quan hệ cha con (Ví dụ: Data Format > Linear Code Identifiers có nghĩa là thẻ cha Data Format chứa thẻ con Linear Code Identifiers). 
Hãy tóm tắt nhóm tính năng này dùng để làm gì.
2. Nếu người dùng CHỈ HỔI về chức năng của một tham số cụ thể: Giải thích dựa trên phần mô tả. ĐỒNG THỜI, hãy chỉ cho người dùng biết tham số này nằm ở vị 
trí nào trong cây cấu hình (thuộc Section nào, nằm trong Panel con nào, Panel cha nào).
3. Nếu người dùng YÊU CẦU ĐIỀU CHỈNH/CÀI ĐẶT/THAY ĐỔI tham số: Bạn CẦN CỐ GẮNG cung cấp 4 thông tin:
   - Tên Config Key (CI_...)
   - Vị trí cài đặt (Nằm trong Panel / Section nào)
   - Mã lệnh (code) (Ví dụ: "04F4")
   - Giá trị cài đặt (value) tương ứng với mục đích của người dùng (tra cứu trong bảng "options").
   * QUAN TRỌNG: Nếu không tìm thấy field 'code' hoặc 'options' phù hợp trong JSON được cung cấp, phải nói rõ "tài liệu không cung cấp mã lệnh này" - TUYỆT ĐỐI KHÔNG tự tạo ra mã hex (code) hoặc bịa đặt giá trị.
4. Trả lời bằng tiếng Việt, tự nhiên, rõ ràng. Chỉ khi nào thông tin HOÀN TOÀN KHÔNG TỒN TẠI mới báo không biết, tuyệt đối không bịa đặt.

CÂU HỎI GỐC CỦA NGƯỜI DÙNG: {user_query}

CÁC TÀI LIỆU CUNG CẤP TỪ HỆ THỐNG:
{full_context}
"""
    response = model.generate_content(prompt)
    return response.text, full_context


def main():
    parser = argparse.ArgumentParser(description='Advanced RAG (Hybrid Search + LLM Router)')
    parser.add_argument('--persist-dir', default='./vectordb_merged', help='Thư mục Chroma DB')
    parser.add_argument('--collection', default='rag4help_merged', help='Tên Collection')
    parser.add_argument('--embedding-model', default='./paraphrase-multilingual-MiniLM-L12-v2', help='Đường dẫn model nhúng')
    parser.add_argument('--top-k', type=int, default=5, help='Số lượng kết quả lấy ra')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🤖 HỆ THỐNG TRỢ LÝ ẢO MÁY QUÉT MÃ VẠCH DATALOGIC ĐÃ SẴN SÀNG")
    print("Gõ 'exit' hoặc 'quit' để thoát chương trình.")
    print("="*70)
    
    while True:
        try:
            # Nhận đầu vào từ Terminal
            user_input = input("\n👤 Bạn: ")
            if not user_input.strip():
                continue
                
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Tạm biệt!")
                break
                
            print("-" * 30)
            
            # Bước 1: LLM Router phân tích câu hỏi
            print("\nĐang xử lý...")
            t0 = time.time()
            
            parsed = parse_query_with_llm(user_input)
            t1 = time.time()
            print(f"   [TIME] Router (Gemini call 1): {t1-t0:.2f}s")
            
            # Bước 2: Tìm kiếm Lai (Hybrid Search)
            results = hybrid_search(
                parsed_query=parsed,
                persist_dir=args.persist_dir,
                collection_name=args.collection,
                embedding_model_path=args.embedding_model,
                top_k=args.top_k
            )
            t2 = time.time()
            print(f"   [TIME] Hybrid search (dense+bm25+rrf): {t2-t1:.2f}s")
            
            # Bước 3: Tổng hợp câu trả lời
            if not results or not results.get("documents"):
                print("Không tìm thấy kết quả nào trong Database.")
                continue
                
            answer, _ = ask_gemini(user_input, results)
            t3 = time.time()
            print(f"   [TIME] Generation (Gemini call 2): {t3-t2:.2f}s")
            print(f"   [TIME] TOTAL TIME: {t3-t0:.2f}s")
            
            print("\n🤖 Trợ lý:")
            print(answer)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f"\nCó lỗi xảy ra: {e}")

if __name__ == '__main__':
    main()
