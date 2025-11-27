#!/usr/bin/env python3
import json
import random
import subprocess
from pathlib import Path
import argparse
import sys
import os

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def run_rag(post, config, device, gen_model):
    cmd = [
        sys.executable,
        "scripts/rag_generate.py",
        "-c", config,
        "--post", post,
        "--device", device,
        "--gen_model", gen_model,
        "--keep", "3",
        "--topk", "60"
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": p.stderr.strip()}
    try:
        return json.loads(p.stdout)
    except Exception:
        return {"error": "Bad JSON output", "raw": p.stdout}

def force_splits_path(filename):
    """Always save inside data/splits/."""
    p = Path("data/splits")
    p.mkdir(parents=True, exist_ok=True)
    return p / filename

def main():
    ap = argparse.ArgumentParser(description="Run RAG on a sample subset of posts.")
    ap.add_argument("--gold", default="data/splits/test_gold.jsonl")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--gen_model", default="google/flan-t5-large")
    ap.add_argument("--sample_size", type=int, default=15)
    ap.add_argument("--input_name", default="sample_rag_input.jsonl")
    ap.add_argument("--output_name", default="sample_rag_output.jsonl")
    args = ap.parse_args()

    gold_path = force_splits_path(Path(args.gold).name)
    input_path = force_splits_path(args.input_name)
    output_path = force_splits_path(args.output_name)

    print(f"Loading gold file → {gold_path}")
    gold_rows = load_jsonl(gold_path)

    print(f"Selecting {args.sample_size} random samples...")
    sample = random.sample(gold_rows, min(args.sample_size, len(gold_rows)))

    # --- SAVE INPUT SAMPLE ---
    write_jsonl(input_path, sample)
    print(f"Saved sampled input posts to → {input_path}")

    # --- RUN RAG ON SAMPLED POSTS ---
    print("\nRunning RAG on sampled posts...\n")
    out_f = open(output_path, "w", encoding="utf-8")

    for i, row in enumerate(sample, 1):
        post = row["post"]
        print(f"=== SAMPLE {i} ===")
        print(f"Post: {post[:200]}...\n")

        result = run_rag(post, args.config, args.device, args.gen_model)

        out_f.write(json.dumps({
            "post": post,
            "rag_output": result
        }, ensure_ascii=False) + "\n")

        print("Reply:")
        if "reply" in result:
            print(result["reply"])
        else:
            print("[NO REPLY] - Error details:")
            print(result)
        print("\n---------------------------")

    out_f.close()
    print(f"\nSaved RAG outputs to → {output_path}")
    print("Done!")

if __name__ == "__main__":
    main()
