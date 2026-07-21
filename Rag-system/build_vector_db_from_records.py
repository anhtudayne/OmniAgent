import argparse
import json
import os
import sys

def ingest_to_chroma(json_dir, persist_dir, collection_name='rag4help_merged', embedding_model='./model', batch_size=64, reset=False):
    # Khai báo import ở đây để nếu máy chưa cài thư viện thì lúc chạy nó mới báo lỗi
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Lỗi: Bạn cần cài đặt chromadb và sentence-transformers trước!")
        sys.exit(1)

    # 1. Tạo thư mục lưu database nếu chưa có
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    # 2. Khởi tạo kết nối với Chroma DB
    client = chromadb.PersistentClient(path=persist_dir)
    
    # 3. Nếu người dùng muốn reset lại collection từ đầu (xóa sạch dữ liệu cũ)
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"Đã xóa collection cũ: {collection_name}")
        except Exception:
            pass

    # 4. Lấy collection ra (hoặc tạo mới nếu chưa có)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={'hnsw:space': 'cosine'}
    )

    print(f"Đang tải model nhúng (Embedding Model): {embedding_model} ...")
    model = SentenceTransformer(embedding_model)

    # Khởi tạo các mảng tạm để gom dữ liệu theo lô (batch)
    pending_ids = []
    pending_documents = []
    pending_metadatas = []
    total_records = 0

    # Hàm phụ để đẩy một lô dữ liệu lên Database
    def flush_batch():
        if len(pending_ids) == 0:
            return 0
            
        print(f"Đang nhúng (embedding) và lưu {len(pending_ids)} bản ghi...")
        
        # Biến đoạn text thành các vector số (embedding)
        embeddings = model.encode(pending_documents, normalize_embeddings=True, show_progress_bar=False).tolist()
        
        # Đẩy vào Chroma DB
        collection.upsert(
            ids=pending_ids,
            documents=pending_documents,
            metadatas=pending_metadatas,
            embeddings=embeddings
        )
        
        # Dọn dẹp mảng tạm để gom lô tiếp theo
        count = len(pending_ids)
        pending_ids.clear()
        pending_documents.clear()
        pending_metadatas.clear()
        
        return count

    # 5. Bắt đầu duyệt qua tất cả các file JSON trong thư mục đầu vào
    for filename in sorted(os.listdir(json_dir)):
        if not filename.endswith('.json'):
            continue
            
        file_path = os.path.join(json_dir, filename)
        print(f"Đang đọc file: {filename}")
        
        # Mở file và load danh sách bản ghi
        with open(file_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
            
        # Duyệt qua từng bản ghi trong file
        for record in records:
            record_id = record.get('id')
            document = record.get('text_content')
            raw_metadata = record.get('metadata')
            
            if raw_metadata is None:
                raw_metadata = {}
                
            # Chú ý: Chroma DB chỉ cho phép metadata là String, Int, Float, Bool.
            # Nên ta dùng vòng lặp For đơn giản để lọc và ép kiểu các trường hợp bị lỗi.
            clean_metadata = {}
            for key, value in raw_metadata.items():
                if value is None:
                    continue
                    
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                else:
                    # Nếu là mảng hay object phức tạp, chuyển thành chuỗi String
                    clean_metadata[key] = str(value)
                    
            # Ta nhét luôn cục raw_json (chứa code, options...) vào metadata dưới dạng String.
            # Điều này giúp lúc truy xuất Vector Search, ta lấy lại được luôn raw_json đưa cho LLM.
            raw_json_data = record.get('raw_json')
            if raw_json_data is not None:
                clean_metadata['raw_json'] = json.dumps(raw_json_data, ensure_ascii=False)
                
            # Đưa bản ghi vào mảng chờ
            pending_ids.append(record_id)
            pending_documents.append(document)
            pending_metadatas.append(clean_metadata)
            total_records = total_records + 1
            
            # Nếu mảng chờ đã đầy (đạt batch_size), thì gọi hàm lưu vào DB
            if len(pending_ids) >= batch_size:
                flush_batch()

    # Nhớ gọi flush_batch lần cuối để lưu nốt các bản ghi còn sót lại (chưa đủ 1 batch)
    flush_batch()
    
    return total_records

def main():
    parser = argparse.ArgumentParser(description='Build Chroma Vector DB from processed RAG records.')
    parser.add_argument('--json-dir', required=True, help='Thư mục chứa các file JSON output (ví dụ: rag_records_output)')
    parser.add_argument('--persist-dir', required=True, help='Thư mục sẽ lưu Chroma DB (ví dụ: vectordb_merged)')
    parser.add_argument('--collection', default='rag4help_merged', help='Tên của Collection (mặc định: rag4help_merged)')
    parser.add_argument('--embedding-model', default='./paraphrase-multilingual-MiniLM-L12-v2', help='Đường dẫn tới folder model (SentenceTransformer)')
    parser.add_argument('--batch-size', type=int, default=64, help='Số lượng bản ghi mỗi lô xử lý (batch_size)')
    parser.add_argument('--reset', action='store_true', help='Nếu truyền tham số này, DB cũ sẽ bị xóa và làm lại từ đầu')
    
    args = parser.parse_args()
    
    print("=== BẮT ĐẦU TẠO VECTOR DATABASE ===")
    
    try:
        total = ingest_to_chroma(
            json_dir=args.json_dir,
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            reset=args.reset
        )
        print(f"=== HOÀN THÀNH: Đã lưu {total} bản ghi vào Vector Database tại '{args.persist_dir}' ===")
        
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
