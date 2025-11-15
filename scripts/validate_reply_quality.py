#!/usr/bin/env python3
import argparse, json, sys, re, statistics
from pathlib import Path
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- IO helpers ----------

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Skipping bad JSONL line {i} in {path}: {e}", file=sys.stderr)
    return rows

def normalize_text(s):
    if s is None:
        return ""
    return str(s).strip()

# ---------- Crisis detection (same spirit as rag_generate) ----------

CRISIS_PATTERNS = [
    r"\bkill myself\b",
    r"\bkill\s+me\b",
    r"\bend my life\b",
    r"\btake my life\b",
    r"\bI (?:want|wanna|feel like) (?:to )?die\b",
    r"\bI (?:want|wanna|feel like) (?:to )?kill myself\b",
    r"\b(?:self[-\s]?harm|hurt myself|overdose)\b",
    r"\bcan['’]?t go on\b",
    r"\bno reason to live\b",
]

def is_crisis_post(text: str) -> bool:
    text = (text or "").lower()
    return any(re.search(p, text) for p in CRISIS_PATTERNS)

def has_crisis_footer(reply: str) -> bool:
    low = (reply or "").lower()
    return (
        "immediate danger" in low and
        ("emergency" in low or "emergency department" in low) and
        ("crisis line" in low or "hotline" in low or "helpline" in low)
    )

# ---------- Structural / heuristic checks ----------

CIT_RE = re.compile(r"\[S\d\]")

def count_sentences(text: str) -> int:
    if not text:
        return 0
    # Very crude: split on ., !, ?
    parts = re.split(r"[\.!?]+", text)
    return sum(1 for p in parts if p.strip())

def reply_metrics(post: str, reply: str, concern: str):
    reply = reply or ""
    concern = (concern or "").strip().lower()

    length_chars = len(reply)
    length_tokens = len(reply.split())
    n_lines = len([ln for ln in reply.splitlines() if ln.strip()])
    n_sents = count_sentences(reply)
    has_citation = bool(CIT_RE.search(reply))

    crisis_flag = (concern == "high") or is_crisis_post(post)
    footer_ok = has_crisis_footer(reply) if crisis_flag else None

    return {
        "length_chars": length_chars,
        "length_tokens": length_tokens,
        "n_lines": n_lines,
        "n_sents": n_sents,
        "has_citation": has_citation,
        "is_crisis": crisis_flag,
        "has_crisis_footer": footer_ok,
    }

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Evaluate reply quality heuristically (semantic similarity + structure).")
    ap.add_argument("--gold", required=True, help="gold jsonl with {post, intent, concern}")
    ap.add_argument("--pred", required=True, help="pred jsonl with {post, reply, ...}")
    ap.add_argument("--encoder_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--top_k_worst", type=int, default=10, help="Show N lowest-similarity examples.")
    args = ap.parse_args()

    gold = load_jsonl(args.gold)
    pred = load_jsonl(args.pred)

    if not gold:
        print("[ERROR] No gold rows loaded.", file=sys.stderr)
        sys.exit(2)
    if not pred:
        print("[ERROR] No pred rows loaded.", file=sys.stderr)
        sys.exit(2)

    # Build pred lookup by post text
    pred_by_post = {}
    for p in pred:
        key = normalize_text(p.get("post", ""))
        if not key:
            continue
        pred_by_post.setdefault(key, []).append(p)

    pairs = []
    missing = 0
    for g in gold:
        post = normalize_text(g.get("post", ""))
        if not post:
            continue
        if post not in pred_by_post:
            missing += 1
            continue
        # If duplicates, just take the first
        p = pred_by_post[post][0]
        pairs.append((g, p))

    if not pairs:
        print("[ERROR] No matched gold/pred pairs (check that 'post' text matches).", file=sys.stderr)
        sys.exit(2)

    print(f"Matched {len(pairs)} gold/pred pairs (missing={missing})")

    # Prepare for semantic similarity
    posts = [normalize_text(g["post"]) for g, _ in pairs]
    replies = [normalize_text(p.get("reply", "")) for _, p in pairs]

    print(f"Loading encoder: {args.encoder_model}", file=sys.stderr)
    encoder = SentenceTransformer(args.encoder_model)
    post_vecs = encoder.encode(posts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    reply_vecs = encoder.encode(replies, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

    sims = (post_vecs * reply_vecs).sum(axis=1)  # cosine, since normalized

    # Collect scalar / Boolean metrics
    lengths = []
    token_counts = []
    line_counts = []
    sent_counts = []
    has_cit_flags = []
    crisis_flags = []
    crisis_footer_flags = []

    examples = []  # for later: (sim, post, reply, concern)

    for (g, p), sim in zip(pairs, sims):
        post = normalize_text(g.get("post", ""))
        concern = normalize_text(g.get("concern", ""))
        reply = normalize_text(p.get("reply", ""))

        m = reply_metrics(post, reply, concern)
        lengths.append(m["length_chars"])
        token_counts.append(m["length_tokens"])
        line_counts.append(m["n_lines"])
        sent_counts.append(m["n_sents"])
        has_cit_flags.append(bool(m["has_citation"]))
        crisis_flags.append(bool(m["is_crisis"]))
        if m["has_crisis_footer"] is not None:
            crisis_footer_flags.append(bool(m["has_crisis_footer"]))

        examples.append((float(sim), post, reply, concern))

    sims_arr = np.array(sims, dtype=float)

    # --------- Aggregate stats ---------
    def pct(x):  # safe percentage helper
        return 100.0 * x if isinstance(x, float) else 100.0 * x

    print("\n=== Semantic similarity (post ↔ reply) ===")
    print(f"Mean similarity: {sims_arr.mean():.3f}")
    print(f"Median similarity: {np.median(sims_arr):.3f}")
    print(f"Min / Max similarity: {sims_arr.min():.3f} / {sims_arr.max():.3f}")
    for thr in [0.30, 0.40, 0.50, 0.60]:
        frac = float((sims_arr < thr).mean())
        print(f"Frac with sim < {thr:.2f}: {frac*100:.1f}%")

    print("\n=== Structural metrics ===")
    print(f"Avg reply length (chars): {statistics.mean(lengths):.1f}")
    print(f"Avg reply length (tokens): {statistics.mean(token_counts):.1f}")
    print(f"Avg #lines: {statistics.mean(line_counts):.2f}")
    print(f"Avg #sentences: {statistics.mean(sent_counts):.2f}")

    n = len(pairs)
    n_short = sum(1 for t in token_counts if t < 15)
    n_long = sum(1 for t in token_counts if t > 120)
    print(f"Replies with <15 tokens: {n_short}/{n} ({n_short/n*100:.1f}%)")
    print(f"Replies with >120 tokens: {n_long}/{n} ({n_long/n*100:.1f}%)")

    has_cit = sum(1 for x in has_cit_flags if x)
    print(f"Replies with at least one [S#] citation: {has_cit}/{n} ({has_cit/n*100:.1f}%)")

    # Crisis coverage
    n_crisis = sum(1 for x in crisis_flags if x)
    if n_crisis > 0:
        n_footer = sum(1 for x in crisis_footer_flags if x)
        print("\n=== Crisis coverage ===")
        print(f"Posts flagged crisis (concern==High or crisis text): {n_crisis}")
        print(f"Replies with crisis footer (immediate danger + emergency + crisis line): "
              f"{n_footer}/{n_crisis} ({n_footer/n_crisis*100:.1f}%)")
    else:
        print("\n=== Crisis coverage ===")
        print("No posts detected as crisis in this split.")

    # --------- Show worst examples by similarity ---------
    examples.sort(key=lambda x: x[0])  # ascending sim
    k = min(args.top_k_worst, len(examples))
    print(f"\n=== {k} lowest-similarity replies (potentially off-topic) ===")
    for i in range(k):
        sim, post, reply, concern = examples[i]
        print(f"\n--- Example {i+1} ---")
        print(f"Similarity: {sim:.3f} | concern={concern}")
        print(f"Post: {post[:300]}{'…' if len(post) > 300 else ''}")
        print(f"Reply: {reply[:300]}{'…' if len(reply) > 300 else ''}")

if __name__ == "__main__":
    main()
