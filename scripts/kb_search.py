#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

import yaml
import faiss
from sentence_transformers import SentenceTransformer


def load_meta(path):
    meta = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                meta.append(json.loads(s))
    return meta


def search(index, encoder, meta, query, intents=None, concern=None, topk=50, keep=5):
    """
    intents: optional set/list of strings (soft OR filter, substring match)
    concern: optional string (exact match after lowering), e.g. 'High'
    """
    # normalize filters
    intents = {i.lower() for i in (intents or [])}
    concern = (concern or "").lower()

    # encode query
    qv = encoder.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, topk)

    # filter + collect
    out = []
    for rank, idx in enumerate(I[0]):
        m = meta[idx]
        ok = True
        if intents:
            ok = any(w in (m.get("intent", "").lower()) for w in intents)
        if ok and concern:
            ok = (m.get("concern", "").lower() == concern)
        if not ok:
            continue
        out.append({
            "rank": len(out) + 1,
            "score": float(D[0][rank]),
            "doc_id": m.get("doc_id", ""),
            "intent": m.get("intent", ""),
            "concern": m.get("concern", ""),
            "text": m.get("text", ""),
            "source": m.get("source", "kb")
        })
        if len(out) >= keep:
            break
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Search the KB (FAISS) with a new post and optional soft filters."
    )
    ap.add_argument("-c", "--config", default="configs/data.yaml",
                    help="Path to YAML config (must contain kb.metadata_jsonl and kb.faiss_index).")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformer encoder (e.g., 'BAAI/bge-small-en-v1.5').")
    ap.add_argument("--query", required=True, help="User post text to search.")
    ap.add_argument("--intent", nargs="*", default=None,
                    help="Soft filter(s) for intent (OR; substring match). Example: --intent 'Seeking Help' Venting")
    ap.add_argument("--concern", default=None,
                    help="Optional concern filter (exact after lowering). Example: --concern High")
    ap.add_argument("--topk", type=int, default=50, help="How many NN to fetch before filtering.")
    ap.add_argument("--keep", type=int, default=5, help="How many results to output after filtering.")
    ap.add_argument("--show-text-len", type=int, default=280, help="Trim snippet text to this many chars in output.")
    ap.add_argument("--export", default=None,
                    help="Optional path to write full JSON results (untrimmed text).")
    args = ap.parse_args()

    # Load config
    cfg = yaml.safe_load(open(args.config))
    kb = cfg.get("kb", {})
    meta_path = kb.get("metadata_jsonl")
    index_path = kb.get("faiss_index")

    if not meta_path or not index_path:
        raise SystemExit("Config must define kb.metadata_jsonl and kb.faiss_index")

    if not (os.path.exists(meta_path) and os.path.exists(index_path)):
        raise SystemExit(f"Missing files. meta={meta_path} index={index_path}")

    # Load artifacts
    meta = load_meta(meta_path)
    index = faiss.read_index(index_path)

    # Optional: improve recall (if stored); otherwise set at runtime
    try:
        index.hnsw.efSearch = max(64, args.topk)
    except Exception:
        pass

    encoder = SentenceTransformer(args.model)

    # Run search
    raw_results = search(
        index=index,
        encoder=encoder,
        meta=meta,
        query=args.query,
        intents=args.intent,
        concern=args.concern,
        topk=args.topk,
        keep=args.keep,
    )

    # Prepare pretty output
    show_n = args.show_text_len
    pretty = {
        "query": args.query,
        "filters": {
            "intent": args.intent,
            "concern": args.concern,
        },
        "results": [
            {
                **{k: v for k, v in r.items() if k != "text"},
                "text": (r["text"][:show_n] + ("..." if len(r["text"]) > show_n else ""))
            }
            for r in raw_results
        ]
    }

    # Print console JSON
    print(json.dumps(pretty, ensure_ascii=False, indent=2))

    # Optional export with full text
    if args.export:
        Path(os.path.dirname(args.export) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump({
                "query": args.query,
                "filters": {"intent": args.intent, "concern": args.concern},
                "results": raw_results  # full, untrimmed text
            }, f, ensure_ascii=False, indent=2)
        print(f"\n[kb_search] wrote full results → {args.export}")


if __name__ == "__main__":
    main()