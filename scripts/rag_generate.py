#!/usr/bin/env python3
"""
RAG generation: retrieve top-K snippets from kb.faiss and generate a grounded reply.

Usage:
  python scripts/rag_generate.py -c configs/data.yaml \
    --post "I can't focus before my exams and I'm panicking." \
    --intents SeekingHelp --concern High --keep 5

SAFETY: This is NOT a replacement for professional mental health care.
All high-risk interactions should be logged and reviewed by trained professionals.
"""

import os
import json
import argparse
import warnings
import re
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# --- safety / stability knobs (set before heavy imports) ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore", message=r"resource_tracker: There appear to be \d+ leaked semaphore")

import yaml
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SAFETY: Crisis Detection
# ============================================================================

def detect_crisis_level(post: str, concern: Optional[str]) -> Tuple[bool, str]:
    """
    Enhanced crisis detection with multiple severity levels.
    Returns: (is_crisis, crisis_level)
    crisis_level: 'immediate', 'high', 'medium', 'none'
    """
    post_lower = post.lower()
    
    # IMMEDIATE DANGER - explicit suicidal ideation or self-harm intent
    immediate_keywords = [
        "kill myself", "end my life", "suicide", "want to die",
        "take my life", "can't go on", "overdose", "gonna jump",
        "going to jump", "planning to", "wrote a note", "saying goodbye"
    ]
    
    # HIGH RISK - self-harm, methods, or passive suicidal ideation
    high_risk_keywords = [
        "cut myself", "cutting myself", "self-harm", "harm myself",
        "better off dead", "everyone would be better", "no reason to live",
        "can't do this anymore", "give up", "hanging", "pills"
    ]
    
    # MEDIUM RISK - distress indicators
    medium_risk_keywords = [
        "hopeless", "worthless", "burden", "pointless", "no point",
        "can't take it", "breaking down", "falling apart"
    ]
    
    # Check immediate danger
    if any(keyword in post_lower for keyword in immediate_keywords):
        return True, "immediate"
    
    # Check concern level override
    if concern and concern.lower() == "high":
        # Verify with keywords
        if any(keyword in post_lower for keyword in high_risk_keywords + immediate_keywords):
            return True, "immediate"
        return True, "high"
    
    # Check high risk
    if any(keyword in post_lower for keyword in high_risk_keywords):
        return True, "high"
    
    # Check medium risk
    if any(keyword in post_lower for keyword in medium_risk_keywords):
        return False, "medium"
    
    return False, "none"


def get_crisis_response(crisis_level: str) -> str:
    """Return appropriate crisis intervention message."""
    if crisis_level == "immediate":
        return (
            "\n\n⚠️ IMMEDIATE SAFETY CONCERN DETECTED ⚠️\n"
            "If you're in immediate danger, please:\n"
            "• Call emergency services (911 in US, 999 in UK, 112 in EU)\n"
            "• Contact a crisis helpline:\n"
            "  - US: 988 Suicide & Crisis Lifeline\n"
            "  - US: Text HOME to 741741 (Crisis Text Line)\n"
            "  - International: https://findahelpline.com\n"
            "• Go to your nearest emergency room\n"
            "• Reach out to a trusted person immediately\n\n"
            "You deserve support from trained professionals who can help keep you safe."
        )
    elif crisis_level == "high":
        return (
            "\n\n⚠️ SAFETY RESOURCES ⚠️\n"
            "What you're experiencing sounds very difficult. Please consider:\n"
            "• Contacting a crisis helpline (988 in US, text HOME to 741741)\n"
            "• Reaching out to a mental health professional\n"
            "• Talking to a trusted friend or family member\n"
            "• If thoughts worsen, seek immediate help\n"
            "Resources: https://findahelpline.com"
        )
    return ""


# ============================================================================
# Data Loading & Retrieval
# ============================================================================

def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file safely."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def filter_unsafe_snippets(snippets: List[Dict]) -> List[Dict]:
    """
    Remove snippets containing explicit self-harm methods or pro-suicide content.
    This is a basic filter - production systems need more sophisticated moderation.
    """
    unsafe_patterns = [
        r'\b(hang|hanging|hanged)\s+(myself|yourself|themselves)',
        r'\b(jump|jumping|jumped)\s+(off|from)',
        r'\bmethod[s]?\s+to\s+(kill|die|suicide)',
        r'\bhow\s+to\s+(kill|die|suicide)',
        r'\b(pills?|medication)\s+(to|and)\s+(die|kill|overdose)',
        r'\bpro-?suicide\b',
        r'\bbetter\s+dead\b.*\bhow\b'
    ]
    
    filtered = []
    for snippet in snippets:
        text_lower = snippet.get("text", "").lower()
        if not any(re.search(pattern, text_lower) for pattern in unsafe_patterns):
            filtered.append(snippet)
        else:
            logger.warning(f"Filtered unsafe snippet: {snippet.get('doc_id', 'unknown')}")
    
    return filtered


def retrieve(
    meta: List[Dict],
    index: faiss.Index,
    encoder: SentenceTransformer,
    post: str,
    intents: Optional[List[str]] = None,
    concern: Optional[str] = None,
    topk: int = 40,
    keep: int = 5,
    min_similarity: float = 0.3
) -> List[Dict]:
    """
    Retrieve and filter snippets with improved scoring.
    
    FIXED: Soft filtering now correctly demotes mismatched items.
    """
    want_intents = {i.lower() for i in (intents or [])}
    want_concern = (concern or "").lower()

    qv = encoder.encode([post], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, topk)

    candidates = []
    for rank, idx in enumerate(I[0]):
        similarity = float(D[0][rank])
        
        # Skip low-similarity results
        if similarity < min_similarity:
            continue
        
        m = meta[idx].copy()
        score = similarity
        
        # FIXED: Demote (not promote) mismatched filters
        if want_intents:
            if not any(w in m.get("intent", "").lower() for w in want_intents):
                score *= 0.85  # Demote by 15%
        
        if want_concern:
            if m.get("concern", "").lower() != want_concern:
                score *= 0.85  # Demote by 15%
        
        candidates.append({
            "rank": rank + 1,
            "score": score,
            "similarity": similarity,
            **m
        })
    
    # Sort by adjusted score and take top K
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_snippets = candidates[:keep]
    
    # Filter unsafe content
    safe_snippets = filter_unsafe_snippets(top_snippets)
    
    # Renumber ranks
    for i, snippet in enumerate(safe_snippets):
        snippet["rank"] = i + 1
    
    return safe_snippets


# ============================================================================
# Prompt Engineering
# ============================================================================

def build_prompt(
    post: str,
    intents: Optional[List[str]],
    concern: Optional[str],
    snippets: List[Dict],
    max_prompt_chars: int = 3000
) -> str:
    """
    Build RAG prompt optimized for Flan-T5.
    
    Flan-T5 works best with:
    - Clear task framing at the start
    - Concise instructions
    - Examples when possible
    - Direct "Answer:" or "Response:" prefix
    """
    
    # Simplified instruction format that Flan-T5 understands better
    instruction = (
        "You are a supportive mental health assistant. Read the user's post and similar examples, "
        "then write a helpful, grounded response (4-6 sentences).\n\n"
    )
    
    # Grounding context - more concise
    context_block = (
        "Here are unrelated examples from past conversations. "
        "They are ONLY for background context. "
        "Do NOT copy them, do NOT refer to them, and do NOT use their content directly.\n"
    )
    for i, s in enumerate(snippets[:3], 1):  # Limit to 3 for token efficiency
        # Truncate snippets more aggressively
        snippet_text = s['text'][:150]
        context_block += f"{i}. {snippet_text}...\n"
    context_block += "\n"
    
    # User post
    post_block = f"User's post:\n{post}\n\n"
    
    # Simplified rules (Flan-T5 responds better to concise directives)
    rules = (
        "Guidelines:\n"
        "- Acknowledge their difficulty\n"
        "- Mention specific details from their post\n"
        "- Suggest 1-2 practical coping steps\n"
        "- Use a calm, neutral tone\n"
        "- Never use: 'I understand how you feel', 'stay strong', 'I've been there'\n"
        "- Do not give medical advice\n\n"
    )
    
    # Simple, direct prompt ending that works with Flan-T5
    task = "Write a supportive response:\n"
    
    prompt = instruction + context_block + post_block + rules +  task
    return prompt[:max_prompt_chars]


# ============================================================================
# Generation & Validation
# ============================================================================

def generate_response(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: str,
    post: str,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 250,
    max_attempts: int = 3
) -> Optional[str]:
    """
    Generate response with improved quality controls and retry logic.
    
    FIXED: Better generation parameters and validation.
    """
    
    # Build bad words list (banned phrases only, not user words)
    banned_phrases = [
        "i understand how you feel", "i've been there", "i can relate",
        "i went through", "stay strong", "you've got this",
        "sending prayers", "god bless", "bless you"
    ]
    
    bad_words_ids = []
    for phrase in banned_phrases:
        try:
            ids = tokenizer(phrase.lower(), add_special_tokens=False).input_ids
            if len(ids) > 1:  # Only ban multi-token phrases
                bad_words_ids.append(ids)
        except Exception:
            pass
    
    # FIXED: Improved generation parameters for Flan-T5
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "min_length": 60,                    # Ensure substantive response
        "max_length": 512,                   # Total budget including prompt
        "length_penalty": 1.5,               # Strong encouragement for complete responses
        "repetition_penalty": 1.2,           # Mild - Flan-T5 is sensitive to high values
        "no_repeat_ngram_size": 3,           # Prevent 3-gram repetition
        "early_stopping": True,
        "num_beams": 4,                      # Beam search for quality
        "do_sample": False,                  # Deterministic by default
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    
    if bad_words_ids:
        gen_kwargs["bad_words_ids"] = bad_words_ids
    
    if do_sample:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": 1
        })
    
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    
    # CRITICAL FIX: Force decoder to start with a proper sentence
    # This prevents it from regurgitating the prompt
    # decoder_start = "The situation"
    # decoder_input_ids = tokenizer(decoder_start, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    # gen_kwargs["decoder_input_ids"] = decoder_input_ids
    
    # Attempt generation with retries
    for attempt in range(max_attempts):
        try:
            gen_ids = model.generate(input_ids, **gen_kwargs)
            reply = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            
            # # Remove decoder start if present
            # if reply.startswith(decoder_start):
            #     reply = reply[len(decoder_start):].strip()
            
            # Validate response
            if is_valid_response(reply, post):
                return reply
            
            logger.warning(f"Attempt {attempt + 1}: Invalid response: {reply[:100]}...")
            
            # For retries, try without forced decoder start
            if attempt == 0 and "decoder_input_ids" in gen_kwargs:
                del gen_kwargs["decoder_input_ids"]
                logger.info("Retry without forced decoder start")
            
            # Add slight randomness for subsequent retries
            if attempt > 0:
                gen_kwargs["temperature"] = 0.8 + (attempt * 0.15)
                gen_kwargs["do_sample"] = True
                gen_kwargs["num_beams"] = 1
                
        except Exception as e:
            logger.error(f"Generation attempt {attempt + 1} failed: {e}")
    
    return None


def is_valid_response(reply: str, post: str) -> bool:
    """
    Validate generated response for quality and safety.
    """
    if not reply or len(reply) < 50:
        logger.warning(f"Response too short: {len(reply)} chars")
        return False
    
    # Must start with capital letter
    if not re.match(r'^[A-Z]', reply):
        logger.warning("Response doesn't start with capital letter")
        return False
    
    # CRITICAL: Check if model is regurgitating prompt instructions
    instruction_markers = [
        "**rules:**", "**structure:**", "**guidelines:**",
        "use neutral, grounded tone",
        "acknowledge the difficulty without",
        "base suggestions on",
        "banned phrases:",
        "do not use first-person",
        "write a supportive response",
        "response:",
        "reply:"
    ]
    reply_lower = reply.lower()
    if any(marker in reply_lower for marker in instruction_markers):
        logger.warning("Response contains prompt instructions")
        return False
    
    # Check for markdown formatting artifacts (shouldn't be in natural response)
    if "**" in reply or reply.count("-") > 5:
        logger.warning("Response contains formatting artifacts")
        return False
    
    # Must contain at least 2 sentences
    sentences = re.split(r'[.!?]+', reply)
    valid_sentences = [s for s in sentences if s.strip() and len(s.strip()) > 10]
    if len(valid_sentences) < 2:
        logger.warning(f"Not enough valid sentences: {len(valid_sentences)}")
        return False
    
    # Should not be too similar to user's post (prevent parroting)
    post_words = set(re.findall(r'\b\w{4,}\b', post.lower()))  # Only words 4+ chars
    reply_words = set(re.findall(r'\b\w{4,}\b', reply.lower()))
    if len(post_words) > 0:
        overlap = len(post_words & reply_words)
        overlap_ratio = overlap / len(post_words)
        if overlap_ratio > 0.7:
            logger.warning(f"Response too similar to user post: {overlap_ratio:.2%} overlap")
            return False
    
    # Check for banned phrases (case-insensitive)
    banned = [
        "i understand how you feel", "i've been there", "i can relate",
        "stay strong", "you've got this", "i know exactly"
    ]
    if any(phrase in reply_lower for phrase in banned):
        logger.warning(f"Response contains banned phrases")
        return False
    
    return True


# ============================================================================
# Logging & Safety
# ============================================================================

def log_interaction(
    post: str,
    concern: Optional[str],
    crisis_level: str,
    reply: str,
    snippets: List[Dict],
    log_dir: str = "logs/interactions"
) -> None:
    """
    Log all interactions for quality monitoring and safety review.
    High-risk interactions should be flagged for human review.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "crisis_level": crisis_level,
        "predicted_concern": concern,
        "post_length": len(post),
        "post_hash": hash(post),  # For anonymization
        "num_snippets": len(snippets),
        "reply_length": len(reply),
        "reply_hash": hash(reply),
        "requires_review": crisis_level in ["immediate", "high"]
    }
    
    # Separate high-risk logs
    if crisis_level in ["immediate", "high"]:
        log_file = os.path.join(log_dir, f"high_risk_{datetime.now().strftime('%Y%m%d')}.jsonl")
    else:
        log_file = os.path.join(log_dir, f"general_{datetime.now().strftime('%Y%m%d')}.jsonl")
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    if crisis_level in ["immediate", "high"]:
        logger.warning(f"HIGH-RISK interaction logged: {crisis_level}")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="RAG: retrieve from KB and generate with Flan-T5.")
    ap.add_argument("-c", "--config", default="configs/data.yaml", help="YAML with kb paths.")
    ap.add_argument("--post", required=True, help="User post text.")
    ap.add_argument("--intents", nargs="*", default=None, help="Optional predicted intents (soft filter).")
    ap.add_argument("--concern", default=None, help="Optional predicted concern (Low/Medium/High).")
    ap.add_argument("--keep", type=int, default=5, help="How many snippets to keep for the prompt.")
    ap.add_argument("--topk", type=int, default=50, help="How many neighbors to fetch before filtering.")
    ap.add_argument("--min_similarity", type=float, default=0.3, help="Minimum similarity threshold.")
    
    # Model args
    ap.add_argument("--enc_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--gen_model", default="google/flan-t5-base",
                    help="Generator model. Consider flan-t5-large for better quality.")

    
    # Generation controls
    ap.add_argument("--max_new_tokens", type=int, default=250)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    
    # Device
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    
    # Logging
    ap.add_argument("--log-dir", default="logs/interactions", help="Directory for interaction logs.")
    ap.add_argument("--no-log", action="store_true", help="Disable interaction logging.")
    
    args = ap.parse_args()
    
    # ========================================================================
    # SAFETY: Crisis Detection First
    # ========================================================================
    is_crisis, crisis_level = detect_crisis_level(args.post, args.concern)
    
    if is_crisis and crisis_level == "immediate":
        # For immediate danger, return crisis resources directly
        crisis_msg = get_crisis_response(crisis_level)
        result = {
            "post": args.post,
            "crisis_detected": True,
            "crisis_level": crisis_level,
            "reply": crisis_msg,
            "disclaimer": "⚠️ This is an automated response. Please seek immediate professional help."
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # Log high-risk interaction
        if not args.no_log:
            log_interaction(args.post, args.concern, crisis_level, crisis_msg, [], args.log_dir)
        
        return
    
    # ========================================================================
    # Load Resources
    # ========================================================================
    try:
        cfg = yaml.safe_load(open(args.config))
        kb = cfg["kb"]
        meta_path = kb["metadata_jsonl"]
        index_path = kb["faiss_index"]
        
        meta = load_jsonl(meta_path)
        index = faiss.read_index(index_path)
        
        # Improve HNSW recall if applicable
        try:
            index.hnsw.efSearch = max(64, args.topk)
        except Exception:
            pass
        
        logger.info(f"Loaded {len(meta)} KB entries from {meta_path}")
        
    except Exception as e:
        logger.error(f"Failed to load resources: {e}")
        print(json.dumps({"error": str(e)}, indent=2))
        return
    
    # ========================================================================
    # Retrieval
    # ========================================================================
    encoder = SentenceTransformer(args.enc_model)
    
    snippets = retrieve(
        meta=meta,
        index=index,
        encoder=encoder,
        post=args.post,
        intents=args.intents,
        concern=args.concern,
        topk=args.topk,
        keep=args.keep,
        min_similarity=args.min_similarity
    )
    
    if not snippets:
        logger.warning("No suitable snippets retrieved. Using fallback response.")
        fallback = (
            "I understand you're going through a difficult time. While I don't have specific "
            "resources to share right now, I encourage you to reach out to a mental health "
            "professional who can provide personalized support. If you're in crisis, please "
            "contact a crisis helpline in your area."
        )
        result = {
            "post": args.post,
            "reply": fallback,
            "citations": [],
            "warning": "No relevant KB entries found"
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    
    logger.info(f"Retrieved {len(snippets)} snippets")
    
    # ========================================================================
    # Generation
    # ========================================================================
    prompt = build_prompt(args.post, args.intents, args.concern, snippets)
    
    tokenizer = AutoTokenizer.from_pretrained(args.gen_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.gen_model, torch_dtype=torch.float32)
    
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    model.to(device)
    logger.info(f"Using device: {device}")
    
    reply = generate_response(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
        post=args.post,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens
    )
    
    if not reply:
        logger.error("Failed to generate valid response after multiple attempts")
        # Construct a basic safe response based on concern level
        if args.concern and args.concern.lower() == "high":
            reply = (
                "What you're describing sounds very difficult and overwhelming. "
                "Given the intensity of what you're experiencing, it would be helpful to speak "
                "with a mental health professional who can provide personalized support. "
                "If you're in crisis, please reach out to a crisis helpline or emergency services."
            )
        else:
            reply = (
                "It sounds like you're going through a challenging time. "
                "While I'm having difficulty providing a specific response right now, "
                "I encourage you to reach out to a mental health professional or trusted person "
                "who can offer personalized support. Your wellbeing matters."
            )
    
    # ========================================================================
    # Add Crisis Footer if Needed
    # ========================================================================
    if is_crisis:
        crisis_footer = get_crisis_response(crisis_level)
        reply += crisis_footer
    
    # ========================================================================
    # Output & Logging
    # ========================================================================
    result = {
        "post": args.post,
        "predicted_intents": args.intents,
        "predicted_concern": args.concern,
        "crisis_detected": is_crisis,
        "crisis_level": crisis_level,
        "citations": [
            {
                "doc_id": s.get("doc_id", ""),
                "intent": s.get("intent", ""),
                "concern": s.get("concern", ""),
                "similarity": s.get("similarity", 0.0)
            }
            for s in snippets
        ],
        "reply": reply,
        "disclaimer": (
            "⚠️ This is an automated support resource, NOT professional mental health care. "
            "For personalized help, please consult a licensed mental health professional."
        )
    }
    
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # Log interaction
    if not args.no_log:
        log_interaction(args.post, args.concern, crisis_level, reply, snippets, args.log_dir)
    
    # Cleanup
    try:
        del encoder, tokenizer, model
        import gc
        gc.collect()
        if device != "cpu":
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()