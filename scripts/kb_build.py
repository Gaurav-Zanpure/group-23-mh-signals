#!/usr/bin/env python3
import os, re, json, argparse
from pathlib import Path

import pandas as pd
import yaml

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def sentish_split(text: str):
    """Lightweight sentence-ish splitter on . ! ? ; keeps punctuation."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = re.split(r"(?<=[\.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(text: str, target_tokens: int = 140, min_chars: int = 20):
    tokens = lambda s: len(s.split())
    sents = sentish_split(text)
    if not sents:
        return []

    chunks, cur, cur_tok = [], [], 0
    for s in sents:
        n = tokens(s)
        if cur and cur_tok + n > target_tokens:
            ch = " ".join(cur).strip()
            if len(ch) >= min_chars:
                chunks.append(ch)
            cur, cur_tok = [], 0
        cur.append(s)
        cur_tok += n

    if cur:
        ch = " ".join(cur).strip()
        if len(ch) >= min_chars:
            chunks.append(ch)

    # If still one huge chunk with no sentence breaks, force split by words
    if len(chunks) == 1 and tokens(chunks[0]) > 2 * target_tokens:
        words, out = chunks[0].split(), []
        for i in range(0, len(words), target_tokens):
            ch = " ".join(words[i:i+target_tokens]).strip()
            if len(ch) >= min_chars:
                out.append(ch)
        chunks = out

    return chunks

def norm_concern(val: str):
    """Normalize Concern_Level to Low/Medium/High if possible."""
    s = (val or "").strip()
    if not s:
        return s
    sl = s.lower()
    # numeric mapping support (0/1/2) or strings
    if sl in {"0", "low"}:
        return "Low"
    if sl in {"1", "med", "medium"}:
        return "Medium"
    if sl in {"2", "high"}:
        return "High"
    return s  # leave as-is if something custom

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/data.yaml")
    ap.add_argument("--csv", help="Override CSV path (else uses paths.raw_csv)")
    args = ap.parse_args()

    # Load YAML
    cfg = yaml.safe_load(open(args.config))
    paths = cfg.get("paths", {})
    kb_cfg = cfg.get("kb", {})

    # Defaults to file/columns if not in YAML
    csv_path = args.csv or paths.get("raw_csv") or "data/raw/kb/rag_intent_data_w_concern.csv"

    processed_dir = kb_cfg.get(
        "processed_dir",
        os.path.join(paths.get("processed_dir", "data/processed"), "kb"),
    )
    corpus_jsonl = kb_cfg.get("corpus_jsonl", os.path.join(processed_dir, "kb_snippets.jsonl"))
    meta_jsonl   = kb_cfg.get("metadata_jsonl", os.path.join(processed_dir, "kb_meta.jsonl"))

    schema = kb_cfg.get("schema", {})
    input_col   = schema.get("input_col",  "input")
    output_col  = schema.get("output_col", "output")
    intent_col  = schema.get("intent_col", "Final_Intent_Tag")
    concern_col = schema.get("concern_col","Concern_Level")
    id_col      = schema.get("id_col",     "doc_id")

    chunking = kb_cfg.get("chunking", {})
    chunk_enabled = bool(chunking.get("enabled", True))
    target_tokens = int(chunking.get("target_tokens", 140))
    min_chars = int(chunking.get("min_chars", 20))

    ensure_dirs(processed_dir)

    print(f"[kb_build] reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Validate required columns
    missing = [c for c in [output_col, intent_col, concern_col] if c not in df.columns]
    if missing:
        raise SystemExit(f"[kb_build] Missing required column(s) in CSV: {missing}")

    # Create doc_id if absent
    if id_col not in df.columns:
        df[id_col] = [f"kb_{i:07d}" for i in range(len(df))]

    # Fill NaNs and cast to str for safety
    for c in [input_col, output_col, intent_col, concern_col, id_col]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    # Normalize concern to consistent labels when possible
    df[concern_col] = df[concern_col].apply(norm_concern)

    # Stable order for reproducibility
    df = df.sort_values(id_col).reset_index(drop=True)

    n_in, n_kept, n_chunks = len(df), 0, 0
    with open(corpus_jsonl, "w", encoding="utf-8") as f_c, open(meta_jsonl, "w", encoding="utf-8") as f_m:
        for _, r in df.iterrows():
            raw_out = (r.get(output_col, "") or "").strip()
            if not raw_out or len(raw_out) < min_chars:
                continue

            base_id = str(r[id_col]).strip()
            pieces = chunk_text(raw_out, target_tokens, min_chars) if chunk_enabled else [raw_out]

            for j, ch in enumerate(pieces, start=1):
                doc_id = f"{base_id}_c{j}" if len(pieces) > 1 else base_id

                # Write corpus (what we embed)
                f_c.write(json.dumps({"doc_id": doc_id, "text": ch}, ensure_ascii=False) + "\n")

                # Write meta (include snippet text so downstream tools don't need a second file lookup)
                meta = {
                    "doc_id": doc_id,
                    "source": "csv",
                    "intent": (r.get(intent_col, "") or "").strip(),
                    "concern": (r.get(concern_col, "") or "").strip(),
                    "tags": "",
                    "text": ch,
                    # "input": (r.get(input_col, "") or "").strip(),
                }
                f_m.write(json.dumps(meta, ensure_ascii=False) + "\n")
                n_chunks += 1

            n_kept += 1

    print(f"[kb_build] rows in CSV: {n_in}")
    print(f"[kb_build] rows kept (output present): {n_kept}")
    print(f"[kb_build] chunks written: {n_chunks}")
    print(f"[kb_build] corpus → {corpus_jsonl}")
    print(f"[kb_build] meta   → {meta_jsonl}")

if __name__ == "__main__":
    main()