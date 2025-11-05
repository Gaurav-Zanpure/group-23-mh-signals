#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/data.yaml")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    paths = cfg.get("paths", {})
    kb = cfg.get("kb", {})
    processed_dir = kb.get("processed_dir", os.path.join(paths.get("processed_dir", "data/processed"), "kb"))

    corpus_jsonl = kb.get("corpus_jsonl", os.path.join(processed_dir, "kb_snippets.jsonl"))
    emb_path     = kb.get("embeddings_npy", os.path.join(processed_dir, "kb_embeddings.npy"))
    faiss_path   = kb.get("faiss_index", os.path.join(processed_dir, "kb.faiss"))

    texts = [row["text"] for row in load_jsonl(corpus_jsonl)]
    if not texts:
        raise SystemExit(f"No texts found in {corpus_jsonl}. Did kb_build.py run?")

    print(f"[kb_encode] loading encoder: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"[kb_encode] encoding {len(texts)} chunks…")
    X = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")

    Path(os.path.dirname(emb_path)).mkdir(parents=True, exist_ok=True)
    np.save(emb_path, X)

    d = X.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 200
    index.add(X)
    faiss.write_index(index, faiss_path)

    print(f"[kb_encode] embeddings → {emb_path} shape={X.shape}")
    print(f"[kb_encode] faiss → {faiss_path}")

if __name__ == "__main__":
    main()