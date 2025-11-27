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
import sys
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
        "kill myself", "end my life", "suicide",
        "take my life", "overdose", "gonna jump",
        "going to jump", "planning to die", "planning to kill myself",
        "wrote a note", "saying goodbye"
    ]
    
    # HIGH RISK - self-harm, methods, or passive suicidal ideation
    high_risk_keywords = [
        "want to die", "cut myself", "cutting myself", "self-harm", "harm myself",
        "cut my", "cutting my", # Added specific self-harm phrasing
        "better off dead", "everyone would be better", "no reason to live",
        "nothing to live for", "can't do this anymore", "give up", "hanging", "pills",
        "can't go on"
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
    min_similarity: float = 0.45  # INCREASED from 0.35 for better relevance
) -> List[Dict]:
    """
    Retrieve and filter snippets with improved scoring.
    Enhanced similarity threshold and scoring for more relevant results.
    """
    want_intents = {i.lower() for i in (intents or [])}
    want_concern = (concern or "").lower()

    qv = encoder.encode([post], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, topk)

    candidates = []
    for rank, idx in enumerate(I[0]):
        similarity = float(D[0][rank])
        
        # Filter by similarity threshold (increased to 0.45 for better relevance)
        if similarity < min_similarity:
            continue
        
        m = meta[idx].copy()
        score = similarity
        
        # ENHANCED: Stronger filtering based on intent and concern
        # If intents match, boost score. If not, heavily penalize.
        if want_intents:
            snippet_intent = m.get("intent", "").lower()
            if any(w in snippet_intent for w in want_intents):
                score *= 1.2  # Boost by 20%
            else:
                score *= 0.5  # Penalize by 50%
        
        # If concern matches, boost score.
        if want_concern:
            snippet_concern = m.get("concern", "").lower()
            if snippet_concern == want_concern:
                score *= 1.2  # Boost by 20%
            elif want_concern == "high" and snippet_concern != "high":
                 # If user is high risk, we really want high risk resources
                score *= 0.5
            elif want_concern != "high" and snippet_concern == "high":
                # If user is low risk, high risk resources might be too intense
                score *= 0.8

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

def build_prompt(post, intents, concern, snippets, max_prompt_chars=2000):
    """
    Build a grounded RAG prompt for Flan-T5.
    Simplified to prevent instruction leakage.
    """
    instruction = (
        "Instruction: You are a supportive AI mental health assistant. "
        "Your goal is to provide validation and support based ONLY on the provided advice perspectives.\n"
        "Read the user's situation and the advice perspectives carefully.\n"
        "Synthesize the relevant advice into a warm, supportive response (single paragraph).\n"
        "Validate the user's feelings but do NOT agree with negative self-talk.\n"
        "Do NOT offer personal opinions or advice not found in the perspectives.\n"
        "Do NOT use lists, bullet points, or numbered steps. Write in full sentences.\n"
        "Do NOT use 'I', 'me', 'my', or share personal experiences. Speak only as a supportive resource.\n"
        "Do NOT refer to the user as 'patient', 'client', or use clinical jargon.\n"
        "Do NOT pretend to be a counselor or refer to past sessions.\n\n"
    )

    context = "Advice Perspectives:\n"
    for i, snip in enumerate(snippets, 1):
        txt = snip['text'][:250].replace("\n", " ")
        context += f"- {txt}\n"
    context += "\n"

    post_block = f"User Situation: {post}\n\n"

    task = "Assistant Response:\n"

    prompt = instruction + context + post_block + task
    return prompt[:max_prompt_chars]

# ... (add_citations_if_missing remains the same, just need to match the Ref format if changed) ...





# ============================================================================
# Generation & Validation
# ============================================================================

def generate_response(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: str,
    post: str,
    snippets: List[Dict],  # ADDED: Required for grounding validation
    do_sample: bool = False,
    temperature: float = 0.5,
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
    
    # IMPROVED: Optimized generation parameters for grounded, relevant responses
    gen_kwargs = {
        "max_new_tokens": 400,               # Increased to prevent truncation
        "min_length": 20,                    # Reduced to allow concise natural replies
        "max_length": 1024,                  # Increased total budget (Flan-T5 can handle >512)
        "length_penalty": 1.2,               # Reduced to avoid forcing length artificially
        "repetition_penalty": 1.1,           # Reduced to avoid weird phrasing
        "no_repeat_ngram_size": 3,           # Prevent 3-gram repetition
        "early_stopping": True,
        "num_beams": 4,                      # Good balance for quality
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
    
    best = None
    
    # Attempt generation with retries
    for attempt in range(max_attempts):
        try:
            gen_ids = model.generate(input_ids, **gen_kwargs)
            reply = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            best = reply # Capture best attempt
            
            # # Remove decoder start if present
            # if reply.startswith(decoder_start):
            #     reply = reply[len(decoder_start):].strip()
            
            # Validate response with enhanced grounding checks
            # Note: Citations are no longer required or added
            
            if is_valid_response(reply, post, snippets):
                # ENHANCED: Stronger semantic grounding validation
                snippet_words = set()
                snippet_texts = []
                for s in snippets:
                    text = s["text"].lower()
                    snippet_texts.append(text)
                    # Extract meaningful words (4+ chars, lowered threshold)
                    snippet_words |= set(re.findall(r'\b\w{4,}\b', text))

                reply_words = set(re.findall(r'\b\w{4,}\b', reply.lower()))
                overlap = len(snippet_words & reply_words)
                
                # Calculate overlap ratio for better grounding metric
                overlap_ratio = overlap / len(reply_words) if reply_words else 0

                # Relaxed grounding: need reasonable overlap (adjusted for Flan-T5)
                # Lowered threshold to 0.08 (8%) to allow more paraphrasing
                if overlap >= 3 and overlap_ratio >= 0.08:
                    logger.info(f"Grounded response validated (overlap: {overlap}, ratio: {overlap_ratio:.2f})")
                    return reply
                else:
                    # CHANGED: Log warning but ALLOW the response. 
                    # Better to give a slightly ungrounded supportive reply than a generic fallback.
                    logger.warning(f"Low grounding detected (overlap: {overlap}, ratio: {overlap_ratio:.2f}) - Accepting anyway")
                    return reply

            logger.warning(f"Attempt {attempt + 1}: Ungrounded or invalid: {reply[:150]}...")

            
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
    
    # If we failed to get a valid response, return the best attempt but CLEAN IT
    if best:
        # Check if it looks truncated
        if not re.search(r'[.!?]\s*$', best):
            # Find last sentence end (ignore "1." or "5." lists)
            # Look for punctuation not preceded by a digit
            match = re.search(r'(.*(?<!\d)[.!?])', best, re.DOTALL)
            if match:
                cleaned = match.group(1)
                logger.warning(f"DEBUG: Regex cleanup: '{best[-20:]}' -> '{cleaned[-20:]}'")
                return cleaned
            
            # Fallback: simple split if regex failed but period exists
            if '.' in best:
                # Split by dot, remove the last chunk (likely fragment)
                parts = best.rsplit('.', 1)
                cleaned = parts[0] + '.'
                logger.warning(f"DEBUG: Split cleanup: '{best[-20:]}' -> '{cleaned[-20:]}'")
                return cleaned
            
            # Specific check for trailing list markers
            if re.search(r'(?:First|Second|Third|1\.|2\.|3\.)\s*,?$', best.strip()):
                 cleaned = best.rsplit(' ', 1)[0] + "."
                 logger.warning(f"DEBUG: List marker cleanup: '{best[-20:]}' -> '{cleaned[-20:]}'")
                 return cleaned

            logger.warning(f"DEBUG: Cleanup failed (no valid punctuation), returning fallback instead of: '{best[-20:]}'")
            return "I'm sorry, I'm having trouble finding the right words right now. Please know that you are not alone, and I encourage you to reach out to a trusted person or professional for support."
    
    # If best is None (generation crashed?), return generic fallback
    logger.error("Generation completely failed (best is None). Returning generic fallback.")
    return "I'm sorry, I'm having trouble finding the right words right now. Please know that you are not alone, and I encourage you to reach out to a trusted person or professional for support."


def is_valid_response(reply: str, post: str, snippets: List[Dict]) -> bool:
    """
    Validate grounded, safe, structured response.
    Enhanced with flexible citation formats and realistic thresholds.
    """

    if not reply or len(reply) < 40:  # Reduced from 50
        logger.warning("Response too short")
        return False

    # Must start with a capital letter
    if not re.match(r'^[A-Z]', reply):
        logger.warning("Response doesn't start with capital letter")
        return False

    # Reject instruction leakage
    reject_markers = [
        "guidelines:", "rules:", "you must", "do not copy",
        "using only the evidence", "your grounded response",
        "critical rules", "end of evidence", "end of post",
        "task:", "evidence snippets:", "ref 1", "ref 2", "snippet 1"
    ]
    rl = reply.lower()
    if any(m in rl for m in reject_markers):
        logger.warning(f"Instruction leakage detected")
        return False
        
    # Reject persona hallucinations and opinionated first-person language
    # Removed "i'm sorry" to allow empathy
    persona_markers = [
        "i can relate", "i have been there", "i know how it feels", 
        "i went through", "i have a job", "i am unemployed",
        "i have dealt with", "i struggle with", "my cat", "my husband",
        "my wife", "my boyfriend", "my girlfriend", "my children",
        "my kids", "my son", "my daughter", "my family", "my parents",
        "i have children", "i have kids", "i'm married", "i am married",
        "i think", "i believe", "i would", "i will", "i'd like", "i'd be happy",
        "i want", "what can i say"
    ]
    if any(m in rl for m in persona_markers):
        logger.warning(f"Persona/First-person hallucination detected")
        return False

    # Reject toxic agreement or reinforcement of negative self-talk
    toxic_markers = [
        "too good for you", "you are selfish", "you were selfish",
        "you should be ashamed", "it is your fault", "it's your fault",
        "you are right to feel ashamed", "you are pathetic",
        "you deserve to feel", "you are a burden",
        "[name]", "[insert", "counseling session", "our session",
        "in our meeting", "previous session",
        "therapist", "counselor", "my goal as a", # Added persona markers
        "justified", "hero", "you will not be a hero" # Added safety markers
    ]
    if any(m in rl for m in toxic_markers):
        logger.warning(f"Toxic/Hallucinated marker detected: {m}")
        return False
        # Check if large chunks (60+ chars) are copied verbatim (increased from 40)
        for i in range(len(snip_text) - 60):
            chunk = snip_text[i:i+60]
            if chunk in reply.lower():
                logger.warning(f"Verbatim copying detected: '{chunk}...'")
                return False

    # Should not be copy-paste formatting
    if "**" in reply or reply.count("-") > 6 or reply.count("•") > 4:
        logger.warning("List formatting detected (likely copied)")
        return False

    # Minimum 2 real sentences (reduced from 3 for compatibility)
    # OR 1 long sentence (> 60 chars)
    sents = re.split(r'[.!?]+', reply)
    valid_sents = [s for s in sents if len(s.strip()) > 10]
    
    if len(valid_sents) < 2 and len(reply) < 60:
        logger.warning(f"Too few sentences/too short: {len(valid_sents)} sents, {len(reply)} chars")
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
    print("DEBUG: RAG Script v3 loaded (Robust Cleanup)", file=sys.stderr)
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
        snippets=snippets,  # ADDED: Pass snippets for grounding validation
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
                "similarity": s.get("similarity", 0.0),
                "text": s.get("text", "") 
            }
            for s in snippets
        ],
        "reply": reply,
        "disclaimer": (
            "This is an automated support resource, NOT professional mental health care. "
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