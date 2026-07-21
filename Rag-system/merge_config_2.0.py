"""
merge_config_2.0.py

Enrich config JSON files (both Magellan and Handheld like Gryphon) with description data:
- For Magellan: uses FRS JSON files in ./data/ directory.
- For Handheld: uses Help JSON files in ./help-json/ directory.

Ensure both output files have exactly the same structure under:
- help_panel_context (panel-level context)
- help_context (parameter-level context)
Any missing fields are set to null.

Usage:
    python merge_config_2.0.py [--input <input_dir>] [--output <output_dir>]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from glob import glob


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_key(s: str) -> str:
    """Normalize a string for fuzzy matching: lowercase, strip, collapse spaces."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


def normalize_handheld_key(s: str) -> str:
    """Normalize a handheld parameter key for matching (replace '.' with ' ')."""
    if not s:
        return ""
    cleaned = s.replace(".", " ")
    return re.sub(r"\s+", " ", cleaned.strip().lower())


def parse_filename_meta(filename: str) -> tuple[str | None, str | None]:
    """Parse model and doc_id/releaseSW from filename: config_Gryphon-GM4500_610099261.json"""
    base = os.path.splitext(filename)[0]
    if base.startswith("config_"):
        parts = base[7:].split("_")
        if len(parts) >= 2:
            return parts[0], parts[1]
        elif len(parts) == 1:
            return parts[0], None
    return None, None


def parse_toc_id(toc_id: str, is_handheld: bool) -> dict:
    """Extract topic, subtopic, model, doc_id from a toc_id string."""
    if not toc_id:
        return {
            "topic": None,
            "subtopic": None,
            "model": None,
            "doc_id": None,
        }

    if is_handheld:
        # e.g., topics_expert_interface_selection_htm -> expert_interface_selection
        rest = toc_id
        if rest.startswith("topics_"):
            rest = rest[7:]
        if rest.endswith("_htm"):
            rest = rest[:-4]
        return {
            "topic": rest or None,
            "subtopic": None,
            "model": None,
            "doc_id": None,
        }
    else:
        # Magellan format parsing
        rest = toc_id[4:] if toc_id.startswith("frs_") else toc_id
        model_pat = re.compile(r'^(Magellan-\d+[a-z]*\d*|XCS-\d+|XMA-\d+)$')
        doc_pat = re.compile(r'^(DR\d+)$')
        dot_idx = rest.find('.')
        if dot_idx == -1:
            topic = rest
            after_dot = ""
        else:
            topic = rest[:dot_idx]
            after_dot = rest[dot_idx + 1:]
        parts = after_dot.split('_') if after_dot else []
        model = None
        doc_id = None
        subtopic_parts = []
        for part in parts:
            if model_pat.match(part):
                model = part
            elif doc_pat.match(part):
                doc_id = part
            else:
                subtopic_parts.append(part)
        subtopic = '_'.join(subtopic_parts) if subtopic_parts else None
        return {
            "topic": topic or None,
            "subtopic": subtopic or None,
            "model": model or None,
            "doc_id": doc_id or None,
        }


def build_panel_context(
    records: list[dict],
    toc_id: str,
    is_handheld: bool,
    filename_model: str | None,
    filename_doc_id: str | None
) -> dict:
    parsed = parse_toc_id(toc_id or "", is_handheld)
    
    topic = None
    subtopic = None
    model = parsed["model"] or filename_model
    doc_id = parsed["doc_id"] or filename_doc_id
    head = None
    source_file = None
    
    if records:
        first = records[0]
        if first.get("topic"):
            topic = first.get("topic")
        if first.get("subtopic"):
            subtopic = first.get("subtopic")
        if first.get("model"):
            model = first.get("model")
        if first.get("doc_id"):
            doc_id = first.get("doc_id")
        if first.get("head"):
            head = first.get("head")
        if first.get("source_file"):
            source_file = first.get("source_file")
            
    return {
        "toc_id": toc_id or None,
        "topic": topic,
        "subtopic": subtopic,
        "model": model,
        "doc_id": doc_id,
        "head": head,
        "source_file": source_file,
        "help_record_count": len(records) if records else 0,
    }


def build_param_context(
    rec: dict | None,
    context_fallback: str | None,
    panel_meta: dict
) -> dict:
    if rec:
        return {
            "parameter_id": rec.get("parameter_id") or None,
            "parameter": rec.get("parameter") or context_fallback or None,
            "description": rec.get("description") or None,
            "notes": rec.get("notes") or None,
            "units": rec.get("units") or None,
            "topic": rec.get("topic") or panel_meta.get("topic") or None,
            "subtopic": rec.get("subtopic") or panel_meta.get("subtopic") or None,
            "model": rec.get("model") or panel_meta.get("model") or None,
            "doc_id": rec.get("doc_id") or panel_meta.get("doc_id") or None,
            "head": rec.get("head") or panel_meta.get("head") or None,
            "section": rec.get("section") or None,
            "tags": rec.get("tags") or None,
            "source_file": rec.get("source_file") or panel_meta.get("source_file") or None,
        }
    else:
        return {
            "parameter_id": None,
            "parameter": context_fallback or None,
            "description": None,
            "notes": None,
            "units": None,
            "topic": panel_meta.get("topic") or None,
            "subtopic": panel_meta.get("subtopic") or None,
            "model": panel_meta.get("model") or None,
            "doc_id": panel_meta.get("doc_id") or None,
            "head": panel_meta.get("head") or None,
            "section": None,
            "tags": None,
            "source_file": panel_meta.get("source_file") or None,
        }


def find_matching_record(
    param_key: str,
    param_data: dict,
    records: list[dict],
    is_handheld: bool
) -> dict | None:
    context = param_data.get("context", "")
    norm_context = normalize_key(context)
    
    if not is_handheld:
        # Magellan exact match strategy
        if not norm_context:
            return None
        for rec in records:
            if normalize_key(rec.get("parameter", "")) == norm_context:
                return rec
        return None
        
    # Gryphon / Handheld matching strategies
    norm_key = normalize_handheld_key(param_key)
    
    # Strategy 1: Exact Match on context or key
    for rec in records:
        norm_rec_param = normalize_key(rec.get("parameter", ""))
        if not norm_rec_param:
            continue
        if norm_context and norm_context == norm_rec_param:
            return rec
        if norm_key and norm_key == norm_rec_param:
            return rec
            
    # Strategy 2: Substring Match
    for rec in records:
        norm_rec_param = normalize_key(rec.get("parameter", ""))
        if not norm_rec_param:
            continue
        if norm_context and (norm_context in norm_rec_param or norm_rec_param in norm_context):
            return rec
        if norm_key and (norm_key in norm_rec_param or norm_rec_param in norm_key):
            return rec
            
    return None


def enrich_panel(
    panel_key: str,
    panel: dict,
    is_handheld: bool,
    filename_model: str | None,
    filename_doc_id: str | None,
    doc_cache: dict,
    help_dir: str,
    data_dir: str,
    stats: dict
) -> None:
    toc_id = panel.get("tocId")
    records = []
    
    if toc_id:
        cache_key = (is_handheld, toc_id)
        if cache_key not in doc_cache:
            if is_handheld:
                # Gryphon/Handheld matching file logic
                if toc_id.startswith("topics_") and toc_id.endswith("_htm"):
                    topic_name = toc_id[7:-4]
                    path = os.path.join(help_dir, f"{topic_name}.json")
                    if os.path.isfile(path):
                        doc_cache[cache_key] = load_json(path)
                    else:
                        doc_cache[cache_key] = []
                else:
                    doc_cache[cache_key] = []
            else:
                # Magellan FRS file logic
                path = os.path.join(data_dir, f"{toc_id}.json")
                if os.path.isfile(path):
                    doc_cache[cache_key] = load_json(path)
                else:
                    doc_cache[cache_key] = []
                    
        records = doc_cache[cache_key]
        if not records:
            print(f"  [WARN] Documentation records not found for tocId: {toc_id} (panel: {panel_key})")
            stats["panels_missing_doc"] += 1
            
    # Build panel context
    panel_context = build_panel_context(records, toc_id, is_handheld, filename_model, filename_doc_id)
    panel["help_panel_context"] = panel_context
    stats["panels_enriched"] += 1
    
    # Walk parameters under panel
    def _walk(obj: dict):
        for key, val in list(obj.items()):
            if not isinstance(val, dict):
                continue
                
            if key.endswith(".pnl"):
                continue
                
            if "type" in val or "value" in val:
                stats["params_total"] += 1
                context = val.get("context", "")
                
                matched_rec = find_matching_record(key, val, records, is_handheld)
                if matched_rec:
                    val["help_context"] = build_param_context(matched_rec, context, panel_context)
                    stats["params_matched"] += 1
                else:
                    val["help_context"] = build_param_context(None, context, panel_context)
                    if context:
                        print(f"    [WARN] No match for context: '{context}' ({key})")
                    stats["params_unmatched"] += 1
                    
                _walk(val)
                continue
                
            _walk(val)
            
    _walk(panel)


def enrich_tree(
    node_key: str,
    node: dict,
    is_handheld: bool,
    filename_model: str | None,
    filename_doc_id: str | None,
    doc_cache: dict,
    help_dir: str,
    data_dir: str,
    stats: dict
) -> None:
    if not isinstance(node, dict):
        return

    if node_key.endswith(".pnl"):
        print(f"  Panel: {node_key}")
        enrich_panel(
            node_key,
            node,
            is_handheld,
            filename_model,
            filename_doc_id,
            doc_cache,
            help_dir,
            data_dir,
            stats
        )

    for child_key, child_value in list(node.items()):
        if isinstance(child_value, dict):
            enrich_tree(
                child_key,
                child_value,
                is_handheld,
                filename_model,
                filename_doc_id,
                doc_cache,
                help_dir,
                data_dir,
                stats
            )


def enrich_config(
    config: dict,
    filename: str,
    help_dir: str,
    data_dir: str
) -> tuple[dict, dict]:
    is_handheld = "Gryphon" in filename
    filename_model, filename_doc_id = parse_filename_meta(filename)
    
    enriched = copy.deepcopy(config)
    doc_cache = {}
    stats = {
        "panels_enriched": 0,
        "panels_missing_doc": 0,
        "params_total": 0,
        "params_matched": 0,
        "params_unmatched": 0,
    }
    
    for root_key, root_value in enriched.items():
        if isinstance(root_value, dict):
            enrich_tree(
                root_key,
                root_value,
                is_handheld,
                filename_model,
                filename_doc_id,
                doc_cache,
                help_dir,
                data_dir,
                stats
            )
            
    return enriched, stats


def main():
    parser = argparse.ArgumentParser(description="Merge configuration parameters with FRS/Help data.")
    parser.add_argument("--input", default="output", help="Input directory containing config JSON files")
    parser.add_argument("--output", default="merged_data4xml", help="Output directory for merged JSON files")
    parser.add_argument("--data", default="data", help="Directory containing FRS JSON files")
    parser.add_argument("--help_dir", default="help-json", help="Directory containing Help JSON files")
    args = parser.parse_args()

    config_files = sorted(glob(os.path.join(args.input, "config_*.json")))
    if not config_files:
        print(f"No JSON files found matching config_*.json in {args.input}/")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    total_panels = 0
    total_missing_panels = 0
    total_params = 0
    total_matched = 0
    total_unmatched = 0

    for config_path in config_files:
        filename = os.path.basename(config_path)
        print(f"\nProcessing: {config_path} (Device Type: {'Handheld/Gryphon' if 'Gryphon' in filename else 'Magellan'})")
        config = load_json(config_path)
        enriched, stats = enrich_config(config, filename, args.help_dir, args.data)

        output_path = os.path.join(args.output, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2, ensure_ascii=False)

        total_panels += stats["panels_enriched"]
        total_missing_panels += stats["panels_missing_doc"]
        total_params += stats["params_total"]
        total_matched += stats["params_matched"]
        total_unmatched += stats["params_unmatched"]

        print(f"  => Output: {output_path}")
        print(
            "  => Stats: "
            f"panels={stats['panels_enriched']} "
            f"(missing_doc={stats['panels_missing_doc']}), "
            f"params={stats['params_total']} matched={stats['params_matched']} unmatched={stats['params_unmatched']}"
        )

    print(f"\n{'='*60}")
    print("Enrichment summary")
    print(f"Panels enriched: {total_panels}")
    print(f"Panels with missing docs: {total_missing_panels}")
    print(f"Parameters total: {total_params}")
    print(f"Parameters matched: {total_matched}")
    print(f"Parameters unmatched: {total_unmatched}")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
