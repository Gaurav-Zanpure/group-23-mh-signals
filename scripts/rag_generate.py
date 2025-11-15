#!/usr/bin/env python3
"""
RAG generation: retrieve top-K snippets from kb.faiss and generate a grounded reply.

Usage:
  python scripts/rag_generate.py -c configs/data.yaml \
    --post "I can't focus before my exams and I'm panicking." \
    --intents SeekingHelp --concern High --keep 5

Notes:
  - Defaults to CPU for stability on macOS. You can try --device mps later.
  - Deterministic by default (no sampling). To enable sampling, pass --do-sample.
"""

import os
import json
import argparse
import warnings

# --- safety / stability knobs (set before heavy imports) ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore", message=r"resource_tracker: There appear to be \d+ leaked semaphore")

import yaml
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieve(meta, index, encoder, post, intents=None, concern=None, topk=40, keep=5):
    """
    Soft-filter retrieval:
      - intents: OR substring match against meta['intent'] (case-insensitive)
      - concern: exact match after lowering
    """
    want_intents = {i.lower() for i in (intents or [])}
    want_concern = (concern or "").lower()

    qv = encoder.encode([post], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, topk)

    out = []
    for rank, idx in enumerate(I[0]):
        m = meta[idx]
        ok = True
        if want_intents:
            ok = any(w in (m.get("intent", "").lower()) for w in want_intents)
        if ok and want_concern:
            ok = (m.get("concern", "").lower() == want_concern)
        if ok:
            out.append({
                "rank": len(out) + 1,
                "score": float(D[0][rank]),
                **m,  # includes: doc_id, text, intent, concern, source (from kb_build.py)
            })
        if len(out) >= keep:
            break
    return out


def build_prompt(post, intents, concern, snippets):
    """
    Flan-T5 style instruction prompting. Keeps the model general, safe, and grounded.
    """
    header = (
        "You are a supportive peer (not a clinician). Be warm, concise (<=140 words), "
        "and base your reply ONLY on the provided snippets. Do NOT diagnose or give "
        "medical instructions. Offer 1–2 coping ideas. If concern is High or the post "
        "suggests self-harm, add a short crisis note at the end.\n\n"
    )
    det = f"User post:\n{post}\n\nDetected: intents={intents}; concern={concern}\n\n"
    cites = "Retrieved snippets:\n" + "\n".join(
        f"[S{i+1}] {s['text']}" for i, s in enumerate(snippets)
    ) + "\n\n"
    task = (
        "Write ONE empathetic reply that:\n"
        "1) Validates feelings\n"
        "2) Gives 1–2 grounded suggestions using [S#]\n"
        "3) Adds a brief crisis note ONLY if appropriate\n\n"
        "Reply:\n"
    )
    return header + det + cites + task


def crisis_footer_needed(post, concern):
    cues = [
        "suicide", "kill myself", "end my life", "self-harm", "overdose",
        "cut myself", "want to die", "take my life", "can't go on"
    ]
    if (concern or "").lower() == "high":
        return True
    low = (post or "").lower()
    return any(t in low for t in cues)


def main():
    ap = argparse.ArgumentParser(description="RAG: retrieve from KB and generate with Flan-T5.")
    ap.add_argument("-c", "--config", default="configs/data.yaml", help="YAML with kb paths.")
    ap.add_argument("--post", required=True, help="User post text.")
    ap.add_argument("--intents", nargs="*", default=None, help="Optional predicted intents (soft filter).")
    ap.add_argument("--concern", default=None, help="Optional predicted concern (Low/Medium/High).")
    ap.add_argument("--keep", type=int, default=5, help="How many snippets to keep for the prompt.")
    ap.add_argument("--topk", type=int, default=40, help="How many neighbors to fetch before filtering.")
    # enc / gen models
    ap.add_argument("--enc_model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Retrieval encoder (e.g., 'BAAI/bge-small-en-v1.5').")
    ap.add_argument("--gen_model", default="google/flan-t5-base",
                    help="Generator model (e.g., 'google/flan-t5-small' for CPU).")
    # generation controls
    ap.add_argument("--max_new_tokens", type=int, default=180)
    ap.add_argument("--do-sample", action="store_true", help="Enable sampling; otherwise deterministic.")
    ap.add_argument("--temperature", type=float, default=0.7, help="Used only if --do-sample is set.")
    ap.add_argument("--top_p", type=float, default=0.9, help="Used only if --do-sample is set.")
    # device
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu",
                    help="Computation device. Start with CPU for stability on macOS.")
    args = ap.parse_args()

    # Load config and artifacts
    cfg = yaml.safe_load(open(args.config))
    kb = cfg["kb"]
    meta_path = kb["metadata_jsonl"]
    index_path = kb["faiss_index"]

    meta = load_jsonl(meta_path)
    index = faiss.read_index(index_path)
    try:
        # improve HNSW recall (safe no-op if not HNSW)
        index.hnsw.efSearch = max(64, args.topk)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Encoder for query
    encoder = SentenceTransformer(args.enc_model)

    # Retrieve
    hits = retrieve(
        meta=meta,
        index=index,
        encoder=encoder,
        post=args.post,
        intents=args.intents,
        concern=args.concern,
        topk=args.topk,
        keep=args.keep,
    )

    # Build prompt
    prompt = build_prompt(args.post, args.intents, args.concern, hits)

    # Generator (Flan-T5)
    tok = AutoTokenizer.from_pretrained(args.gen_model)
    # Force float32 on CPU for stability
    mdl = AutoModelForSeq2SeqLM.from_pretrained(args.gen_model, torch_dtype=torch.float32)

    device = "cpu"
    if args.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    mdl.to(device)

    in_ids = tok(prompt, return_tensors="pt", truncation=True).input_ids.to(device)

    # Deterministic by default (no temperature warning, no sampling)
    if args.do_sample:
        gen_ids = mdl.generate(
            in_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        gen_ids = mdl.generate(in_ids, max_new_tokens=args.max_new_tokens)

    reply = tok.decode(gen_ids[0], skip_special_tokens=True)

    # Crisis footer (safety)
    if crisis_footer_needed(args.post, args.concern) and "immediate danger" not in reply.lower():
        reply += (
            "\n\nIf you’re in immediate danger, please contact local emergency services or a trusted person nearby. "
            "If you can, consider reaching out to a crisis line in your region."
        )

    # Output JSON (includes citations)
    print(json.dumps({
        "post": args.post,
        "predicted_intents": args.intents,
        "predicted_concern": args.concern,
        "citations": [
            {"doc_id": h.get("doc_id", ""), "intent": h.get("intent", ""), "concern": h.get("concern", "")}
            for h in hits
        ],
        "reply": reply
    }, ensure_ascii=False, indent=2))

    # tidy up
    try:
        del encoder, tok, mdl
        import gc
        gc.collect()
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
