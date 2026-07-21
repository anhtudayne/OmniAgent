import json
import os
import argparse
import hashlib

def process_file(file_path):
    filename = os.path.basename(file_path)
    
    # Bước 1: Bóc tách Metadata từ Tên File (File-Level Metadata)
    base_name = filename.replace('.json', '')
    parts = base_name.split('_')
    
    file_metadata = {}
    if len(parts) >= 3:
        file_metadata = {
            "file_type": parts[0],        # config
            "model": parts[1],            # Magellan-900i, Magellan-1500i, hay Gryphon-GM4500
            "doc_id": parts[2],           # DR9401800, DR9401843, hay 610099261
            "topic": "_".join(parts[3:]) if len(parts) >= 4 else "Unknown"
        }
    else:
        file_metadata = {
            "file_type": "unknown",
            "model": "unknown",
            "doc_id": "unknown",
            "topic": base_name
        }

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = []
    
    # BƯỚC 1: Tiền xử lý để xây dựng cây gia phả (Hierarchy) của các Panel
    id_to_title = {}
    id_to_parent = {}
    
    for key, value in data.items():
        if key.endswith('.pnl') and isinstance(value, dict):
            parts = key[:-4].split('.')
            child_id = parts[-1]
            title = value.get('title', child_id)
            
            id_to_title[child_id] = title
            # Nếu key có dạng Parent.Child.pnl
            if len(parts) > 1:
                id_to_parent[child_id] = parts[-2]
                
    def get_panel_path_from_key(key, title):
        """Hàm đệ quy ngược lên cha để lấy chuỗi đường dẫn (Breadcrumb), lấy trực tiếp parent từ key để tránh ghi đè ở node lá"""
        parts = key[:-4].split('.')
        if len(parts) == 1:
            return title
            
        parent_id = parts[-2]
        path = []
        curr = parent_id
        while curr:
            path.insert(0, id_to_title.get(curr, curr))
            curr = id_to_parent.get(curr)
            
        path.append(title)
        return " > ".join(path)

    # BƯỚC 2: Duyệt đệ quy để gom CI_
    def walk_node(node, current_panel_path):
        if not isinstance(node, dict):
            return
            
        for key, value in node.items():
            if not isinstance(value, dict):
                continue
                
            # Chấp nhận cả CI_ bắt đầu (Magellan) và chứa .CI_ (Handheld như Gryphon)
            if key.startswith('CI_') or '.CI_' in key:
                record = process_ci_node(key, value, current_panel_path, file_metadata)
                if record:
                    records.append(record)
            elif key.endswith('.pnl'):
                # Đệ quy vào panel con
                title = value.get('title', key[:-4].split('.')[-1])
                full_path = get_panel_path_from_key(key, title)
                walk_node(value, full_path)
            else:
                # Đệ quy vào các node dictionary khác
                walk_node(value, current_panel_path)

    def process_ci_node(ci_key, ci_val, panel_path_str, file_meta):
        # Hỗ trợ cả help_context (từ file merge 2.0) và frs_context (từ file merge frs cũ)
        doc_context = ci_val.get('help_context') or ci_val.get('frs_context') or {}
        
        # Bước 3: Trích xuất Metadata Cục bộ (Item-Level Metadata)
        parameter_id = doc_context.get('parameter_id', ci_key)
        section = doc_context.get('section', 'Unknown')
        
        item_metadata = {
            "panel_name": panel_path_str or 'Unknown',
            "parameter_id": parameter_id,
            "ci_key": ci_key,
            "section": section
        }
            
        # Nếu tên file không có topic (bị gán là Unknown), lấy bổ sung từ context
        if file_meta.get("topic") == "Unknown":
            item_metadata["topic"] = doc_context.get("topic", "Unknown")
        
        # Bước 4: Tạo "Template Văn bản" (Text for Embedding)
        text_lines = []
        text_lines.append(f"Config Key: {ci_key}")
        text_lines.append(f"Parameter: {doc_context.get('parameter', ci_key)}")
        text_lines.append(f"Panel: {item_metadata['panel_name']}")
        text_lines.append(f"Section: {item_metadata['section']}")
        
        description = doc_context.get('description')
        if description:
            text_lines.append(f"Description: {description}")
            
        notes = doc_context.get('notes')
        if notes:
            text_lines.append(f"Notes: {notes}")
            
        options = ci_val.get('options')
        if options:
            opt_texts = [f"{k}: {v}" for k, v in options.items()]
            text_lines.append(f"Options: {', '.join(opt_texts)}")
            
        min_val = ci_val.get('min')
        max_val = ci_val.get('max')
        if min_val is not None or max_val is not None:
            text_lines.append(f"Range: Min {min_val}, Max {max_val}")
            
        text_content = "\n".join(text_lines)
        
        # Bước 5: Cô lập JSON Cục bộ (Mini-JSON for LLM)
        keys_to_keep = ['value', 'type', 'code', 'sizeLen', 'tableRef', 'options', 'interfaceDefaults', 'min', 'max', 'incrementBy', 'range', 'protection']
        mini_json = {k: ci_val[k] for k in keys_to_keep if k in ci_val}
        mini_json['ci_key'] = ci_key
        
        # Bước 6: Đóng gói Bản ghi (Final Document Assembly)
        merged_metadata = {**file_meta, **item_metadata}
        
        # Tạo ID duy nhất (Tên máy + Mã tài liệu + Mã CI + Mã Hash của nội dung)
        model = file_meta.get("model", "unknown")
        doc_id = file_meta.get("doc_id", "unknown")
        
        # Băm nội dung để chống trùng lặp ID nếu một tham số xuất hiện nhiều lần ở các Panel khác nhau
        content_hash = hashlib.sha1(text_content.encode('utf-8')).hexdigest()[:8]
        record_id = f"{model}_{doc_id}_{ci_key}_{content_hash}"
            
        record = {
            "id": record_id,
            "metadata": merged_metadata,
            "text_content": text_content,
            "raw_json": mini_json
        }
        return record

    walk_node(data, "")
    return records

def main():
    parser = argparse.ArgumentParser(description="Process merged config JSON into RAG records.")
    parser.add_argument("--input-dir", required=True, help="Directory containing the input JSON files.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the output JSON files.")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} not found.")
        return
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    total_files = 0
    total_records = 0
    
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(args.input_dir, filename)
            print(f"Processing {filename}...")
            records = process_file(file_path)
            
            # Ghi ra file JSON riêng lẻ
            output_file_path = os.path.join(args.output_dir, filename)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
                
            print(f" -> Saved {len(records)} records to {output_file_path}")
            total_files += 1
            total_records += len(records)
            
    print(f"Successfully processed {total_files} files with a total of {total_records} records.")

if __name__ == "__main__":
    main()
