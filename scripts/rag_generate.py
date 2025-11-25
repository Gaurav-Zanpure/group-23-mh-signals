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
import re

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
            if not any(w in m.get("intent", "").lower() for w in want_intents):
                # only demote score, don't drop the snippet
                m["score"] *= 1.05  
        if want_concern:
            if m.get("concern", "").lower() != want_concern:
                # softly downrank, don't drop
                m["score"] *= 1.05
        if ok:
            out.append({
                "rank": len(out) + 1,
                "score": float(D[0][rank]),
                **m,  # includes: doc_id, text, intent, concern, source (from kb_build.py)
            })
        if len(out) >= keep:
            break
    return out


def build_prompt(post, intents, concern, snippets, max_prompt_chars=2000):
    """
    SAFE RAG prompt: Model sees only ABSTRACT emotional signals.
    Prevents sympathy clichés, generic advice, or repetition.
    
    FIX: Added crucial instruction to prevent first-person phrases like 'I can relate'.
    """

    # # Abstract emotional labels (never direct words)
    # def abstract_theme(text):
    #     t = text.lower()
    #     label = []

    #     if any(w in t for w in ["hurt", "pain", "break", "heart", "cry"]):
    #         label.append("intense emotional struggle")
    #     if any(w in t for w in ["anxiety", "panic", "fear"]):
    #         label.append("inner tension")
    #     if any(w in t for w in ["lost", 'confused', 'overwhelmed']):
    #         label.append("uncertainty")
    #     if any(w in t for w in ["alone", "lonely", "isolated"]):
    #         label.append("feeling disconnected")
    #     if any(w in t for w in ["guilt", "regret", "blame"]):
    #         label.append("self-judgment")
    #     if any(w in t for w in ["support", "help", "cope"]):
    #         label.append("desire for relief")
    #     if any(w in t for w in ["trauma", "ptsd"]):
    #         label.append("distress from past experiences")

    #     if not label:
    #         return "general inner difficulty"

    #     # Return single abstract signal (prevents list-like tone)
    #     return " and ".join(label[:1])

    # themes = [abstract_theme(s["text"]) for s in snippets]

    # # Instead of listing themes → blend them into 1 abstract signal
    # combined_emotion = ", ".join(set(themes))

    # USER POST FIRST (important)
    post_block = f"User Post:\n{post}\n\n"

    # Abstract emotional signal (NO LISTS, NO BULLETS)
    # theme_block = (
    #     f"Underlying emotional cues from similar situations (use as intuition only): "
    #     f"{combined_emotion}.\n\n"
    # )

    # GROUNDING CONTEXT (Retained Snippets - Crucial)
    # The model will use the text of these similar posts for factual grounding.
    context_block = "The following snippets are from similar experiences and are provided for factual grounding only. Do NOT cite them directly. Use them to inform the tone and suggestions:\n"
    for s in snippets:
        context_block += f"- {s['text']}\n"
    context_block += "\n"

    # instructions = (
    #     "Write a concise, emotionally-safe response in **3–4 sentences**. "
    #     "Use a neutral, grounded tone – not warm, intimate, or overly reassuring. "
    #     "Do NOT say: 'I understand', 'I know how you feel', 'I'm sorry', 'you are not alone', "
    #     "'stay strong', or any other empathy cliché, reassurance, or encouragement. "
    #     "Do NOT use first person to talk about yourself. "
    #     "Do NOT give clinical, diagnostic, or medical advice. "
    #     "\n\n"

    #     "Your reply must follow this structure:"
    #     "\n"
    #     "1) **Acknowledge** the user's experience without assuming their feelings and without using 'I'. \n"
    #     "2) Describe what seems difficult based on the specific details in the post. \n"
    #     "3) If risk is LOW or MEDIUM: offer 1-2 gentle, practical next steps that may help them cope. \n"
    #     "4) If the post suggests self-harm or danger: do NOT give suggestions; instead gently encourage reaching out to emergency services or a crisis hotline. \n"
    #     "\n\n"

    #     "Use calm, clear, specific language. Avoid emotional clichés. Focus on the facts of the user's experience. "
    #     "Do NOT give emotional clichés, comfort phrases, or personal sharing. "
    #     "\n\nReply:\n"
    # )
    # Revised Instructions
    instructions = (
        "You are a helpful and neutral mental health resource, not a therapist. "
        "Your task is to provide a grounded, emotionally safe response. "
        "\n\n"

        # "Always begin your reply with a complete, natural-sounding first sentence starting with a capital letter. "
        "Never start a partial word, suffix, dash, or anything that is not a complete sentence. "
        "Always base your reply ONLY on the provided snippets. "
        "\n\n"

        "**Instructions for Reply (Aim for 3-5 Sentences):**"
        "\n"
        "1) **Acknowledge and Validate:** Begin by recognizing the difficulty of the situation described. Use neutral language (e.g., 'The experience of X sounds incredibly challenging'). \n"
        "2) **Ground in Details:** Briefly reference the specific factual elements of the post (e.g., 'Dealing with grief, academic stress, and social rejection all at once'). \n"
        "3) **Offer Practical Next Steps:** Suggest 1-2 gentle, practical steps focused on coping, self-care, or accessing support. \n"
        "4) **Maintain a Neutral Persona:** Absolutely do NOT use first person to talk about yourself, share personal stories, give reassurance clichés, or use overly warm/intimate language. Do NOT offer clinical or diagnostic advice."
        "\n\n"

        "**Banned Phrases (Do not use):** I understand, I know how you feel, I can relate, I'm sorry to hear that, I've been there, I went through, I'm here for you, stay strong, you've got this, praying for you, I can relate to."
        "\n\n"

        "Reply:\n"
    )

    forced_start_phrase = "Response: The experience you are describing"

    prompt = instructions + context_block + post_block + forced_start_phrase
    return prompt[:max_prompt_chars].strip()


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
    ap.add_argument("--keep", type=int, default=3, help="How many snippets to keep for the prompt.")
    ap.add_argument("--topk", type=int, default=40, help="How many neighbors to fetch before filtering.")
    # enc / gen models
    ap.add_argument("--enc_model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Retrieval encoder (e.g., 'BAAI/bge-small-en-v1.5').")
    ap.add_argument("--gen_model", default="google/flan-t5-base",
                    help="Generator model (e.g., 'google/flan-t5-small' for CPU).")
    # generation controls
    ap.add_argument("--max_new_tokens", type=int, default=200)
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

    # --------------------- HIGH-QUALITY GENERATION SETTINGS ---------------------
    gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "min_length": 40,                    # enforce at least 4–6 sentences
            "max_length": 400,                   # total token budget including input
            "length_penalty": 1.0,               # encourages longer replies
            "repetition_penalty": 3.0,
            "no_repeat_ngram_size": 4,
            "early_stopping": True,
            "num_beams": 4,
            "do_sample": False,
            # FIX: Increased from 1.1 to 1.3 to more strongly penalize paraphrasing of the input post
            "encoder_repetition_penalty": 2.5, 
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.eos_token_id,
    }

    if args.do_sample:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_beams": 1
        })

    # --- FIX: Enforce prompt instructions by banning cliches and first-person phrases ---
    
    # 1. Ban user's own words (long words only)
    bad_words = []
    user_words = re.findall(r"\b\w+\b", args.post.lower())
    for w in user_words:
        if len(w) > 4:
            # Tokenize and add as a sequence to ban
            try:
                bad_words.append(tok(w, add_special_tokens=False).input_ids)
            except Exception:
                pass
    
    # 2. Block sympathy clichés and banned first-person phrases (from prompt instructions)
    banned_phrases = [
        "i understand", "i know how you feel", "i can relate",
        "i've been there", "i went through", "i'm sorry to hear that",
        "stay strong", "you've got this", "keep going", "it will get better",
        "i love you", "love you", "praying for you", "bless you", "god bless",
        "i'm proud of you", "i'm happy for you",
    ]
    
    for banned in banned_phrases:
        try:
            # Tokenize each phrase into its components and add the list of token IDs
            banned_token_ids = tok(banned.lower(), add_special_tokens=False).input_ids
            # Only ban multi-token phrases to avoid over-blocking common single tokens (like 'I' or 'to')
            if len(banned_token_ids) > 1:
                bad_words.append(banned_token_ids)
        except Exception:
            pass

    if bad_words:
        gen_kwargs["bad_words_ids"] = bad_words
    # ----------------------------------------------------------------------------------

    gen_ids = mdl.generate(in_ids, **gen_kwargs)
    # ---------------------------------------------------------------------------

    reply = tok.decode(gen_ids[0], skip_special_tokens=True)

    # Crisis footer (safety)
    if crisis_footer_needed(args.post, args.concern) and "immediate danger" not in reply.lower():
        reply += (
            "\n\nIf you’re in immediate danger, please contact local emergency services or a trusted person nearby. "
            "If you can, consider reaching out to a crisis line in your region."
        )
    

    def needs_regeneration(text):
        t = text.strip()
        if not t:
            return True
        # Must begin with A-Z (English clean sentence start)
        if not re.match(r"[A-Z]", t[0]):
            return True
        return False

    if needs_regeneration(reply):
        gen_ids = mdl.generate(in_ids, **gen_kwargs)
        reply = tok.decode(gen_ids[0], skip_special_tokens=True).strip()

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