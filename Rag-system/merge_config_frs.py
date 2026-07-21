"""
Enrich config spec JSON (data4xml) with FRS description JSON (data) using tocId linkage.

The output keeps the original data4xml structure intact and appends contextual fields
from FRS under each panel and CI parameter node.

Usage:
    python merge_config_frs.py
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from glob import glob

DATA4XML_DIR = "data4xml"
DATA_DIR = "data"
OUTPUT_DIR = "merged_data4xml"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_frs_file(toc_id: str) -> list[dict] | None:
    """Find the FRS JSON file matching a tocId in ./data/."""
    path = os.path.join(DATA_DIR, f"{toc_id}.json")
    if os.path.isfile(path):
        return load_json(path)
    return None


def normalize_key(s: str) -> str:
    """Normalize a string for fuzzy matching: lowercase, strip, collapse spaces."""
    return re.sub(r"\s+", " ", s.strip().lower())

def parse_toc_id(toc_id: str) -> dict:
    """Extract topic, subtopic, model, doc_id from a toc_id string.
    toc_id format examples:
      frs_codeselection.linearcodeselection
      frs_linearcodeselection.codeeanupc_Magellan-900i_DR9401636
    Returns a dict with keys: topic, subtopic, model, doc_id.
    """
    # Strip leading 'frs_' prefix if present
    rest = toc_id[4:] if toc_id.startswith("frs_") else toc_id
    model_pat = re.compile(r'^(Magellan-\d+[a-z]*\d*|XCS-\d+|XMA-\d+)$')
    doc_pat = re.compile(r'^(DR\d+)$')
    # Split on '.' to separate topic from subtopic+extras
    dot_idx = rest.find('.')
    if dot_idx == -1:
        # No dot: entire string is the topic
        topic = rest
        after_dot = ""
    else:
        topic = rest[:dot_idx]
        after_dot = rest[dot_idx + 1:]
    # after_dot may be: "linearcodeselection" or "codeeanupc_Magellan-900i_DR9401636"
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
        "topic": topic,
        "subtopic": subtopic,
        "model": model,
        "doc_id": doc_id,
    }


def build_frs_lookup(frs_records: list[dict]) -> dict[str, dict]:
    """Build a lookup dict: normalized parameter name -> FRS record."""
    lookup = {}
    for rec in frs_records:
        key = normalize_key(rec.get("parameter", ""))
        if key:
            lookup[key] = rec
    return lookup


def build_panel_context(frs_records: list[dict], toc_id: str) -> dict:
    if not frs_records:
        # File exists but is empty (or file not found): derive metadata from toc_id itself
        parsed = parse_toc_id(toc_id)
        return {
            "toc_id": toc_id,
            "topic": parsed["topic"],
            "subtopic": parsed["subtopic"],
            "model": parsed["model"],
            "doc_id": parsed["doc_id"],
            "head": "",
            "source_file": "",
            "frs_record_count": 0,
        }
    first = frs_records[0]
    return {
        "toc_id": toc_id,
        "topic": first.get("topic", ""),
        "subtopic": first.get("subtopic", ""),
        "model": first.get("model", ""),
        "doc_id": first.get("doc_id", ""),
        "head": first.get("head", ""),
        "source_file": first.get("source_file", ""),
        "frs_record_count": len(frs_records),
    }

def build_param_context(frs_match: dict, context_fallback: str) -> dict:
    return {
        "parameter_id": frs_match.get("parameter_id", ""),
        "parameter": frs_match.get("parameter", context_fallback),
        "description": frs_match.get("description", ""),
        "notes": frs_match.get("notes", ""),
        "units": frs_match.get("units", ""),
        "topic": frs_match.get("topic", ""),
        "subtopic": frs_match.get("subtopic", ""),
        "model": frs_match.get("model", ""),
        "doc_id": frs_match.get("doc_id", ""),
        "head": frs_match.get("head", ""),
        "section": frs_match.get("section", ""),
        "tags": frs_match.get("tags", ""),
        "source_file": frs_match.get("source_file", ""),
    }


def _empty_param_context(context_fallback: str) -> dict:
    return {
        "parameter_id": "",
        "parameter": context_fallback,
        "description": "",
        "notes": "",
        "units": "",
        "topic": "",
        "subtopic": "",
        "model": "",
        "doc_id": "",
        "head": "",
        "section": "",
        "tags": "",
        "source_file": "",
    }


def enrich_ci_nodes_in_panel(
    node: dict,
    frs_lookup: dict[str, dict],
    stats: dict,
) -> tuple[int, int]:
    """
    Recursively enrich all CI_* nodes inside one panel.
    Important: do not cross into nested .pnl nodes, since they have their own tocId.
    """
    ci_total = 0
    ci_matched = 0

    def _walk(obj: dict):
        nonlocal ci_total, ci_matched

        for key, val in obj.items():
            if not isinstance(val, dict):
                continue

            if key.endswith(".pnl"):
                # Nested panel will be handled by its own enrich_panel() call.
                continue

            if key.startswith("CI_"):
                ci_total += 1
                context = val.get("context", "")
                frs_match = frs_lookup.get(normalize_key(context)) if context and frs_lookup else None

                if frs_match is None:
                    val["frs_context"] = _empty_param_context(context)
                    if context:
                        print(f"    [WARN] No FRS match for context: '{context}' ({key})")
                    stats["params_unmatched"] += 1
                else:
                    val["frs_context"] = build_param_context(frs_match, context)
                    ci_matched += 1
                    stats["params_matched"] += 1

                # Continue walking this CI node to catch accidentally nested CI_* keys.
                _walk(val)
                continue

            _walk(val)

    _walk(node)
    return ci_total, ci_matched


def enrich_panel(panel_key: str, panel: dict, frs_cache: dict[str, list[dict]], stats: dict) -> None:
    """
    Enrich a panel dict in-place while preserving all existing keys/values.
    - Adds panel-level field: `frs_panel_context`
    - Adds parameter-level field for each CI_*: `frs_context`
    """
    toc_id = panel.get("tocId")
    frs_records = []
    frs_lookup = {}

    if toc_id:
        if toc_id not in frs_cache:
            frs_cache[toc_id] = find_frs_file(toc_id) or []
        frs_records = frs_cache[toc_id]

        if not frs_records:
            print(f"  [WARN] FRS file not found for tocId: {toc_id} (panel: {panel_key})")
            stats["panels_missing_frs"] += 1
        else:
            frs_lookup = build_frs_lookup(frs_records)

    panel["frs_panel_context"] = build_panel_context(frs_records, toc_id or "")
    stats["panels_enriched"] += 1

    ci_total, ci_matched = enrich_ci_nodes_in_panel(panel, frs_lookup, stats)

    stats["params_total"] += ci_total
    if ci_total:
        print(f"    Matched: {ci_matched}/{ci_total} parameters")


def enrich_tree(node_key: str, node: dict, frs_cache: dict[str, list[dict]], stats: dict) -> None:
    """Recursively enrich any object node that represents a .pnl panel."""
    if not isinstance(node, dict):
        return

    if node_key.endswith(".pnl"):
        print(f"  Panel: {node_key}")
        enrich_panel(node_key, node, frs_cache, stats)

    for child_key, child_value in list(node.items()):
        if isinstance(child_value, dict):
            enrich_tree(child_key, child_value, frs_cache, stats)


def enrich_config(config: dict) -> tuple[dict, dict]:
    """Return enriched config plus summary stats."""
    enriched = copy.deepcopy(config)
    frs_cache: dict[str, list[dict]] = {}
    stats = {
        "panels_enriched": 0,
        "panels_missing_frs": 0,
        "params_total": 0,
        "params_matched": 0,
        "params_unmatched": 0,
    }

    for root_key, root_value in enriched.items():
        if isinstance(root_value, dict):
            enrich_tree(root_key, root_value, frs_cache, stats)

    return enriched, stats


def main():
    # Find all config JSON files in data4xml
    config_files = sorted(glob(os.path.join(DATA4XML_DIR, "*.json")))
    if not config_files:
        print(f"No JSON files found in {DATA4XML_DIR}/")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_panels = 0
    total_missing_panels = 0
    total_params = 0
    total_matched = 0
    total_unmatched = 0

    for config_path in config_files:
        print(f"\nProcessing: {config_path}")
        config = load_json(config_path)
        enriched, stats = enrich_config(config)

        output_name = os.path.basename(config_path)
        output_path = os.path.join(OUTPUT_DIR, output_name)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2, ensure_ascii=False)

        total_panels += stats["panels_enriched"]
        total_missing_panels += stats["panels_missing_frs"]
        total_params += stats["params_total"]
        total_matched += stats["params_matched"]
        total_unmatched += stats["params_unmatched"]

        print(f"  => Output: {output_path}")
        print(
            "  => Stats: "
            f"panels={stats['panels_enriched']} "
            f"(missing_frs={stats['panels_missing_frs']}), "
            f"params={stats['params_total']} matched={stats['params_matched']} unmatched={stats['params_unmatched']}"
        )

    print(f"\n{'='*60}")
    print("Enrichment summary")
    print(f"Panels enriched: {total_panels}")
    print(f"Panels with missing FRS: {total_missing_panels}")
    print(f"Parameters total: {total_params}")
    print(f"Parameters matched: {total_matched}")
    print(f"Parameters unmatched: {total_unmatched}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
