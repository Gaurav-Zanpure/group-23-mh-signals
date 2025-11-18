#!/usr/bin/env python3
import argparse, json, subprocess, sys, time, re
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
RAG = HERE / "rag_generate.py"


def truncate_post(post: str, max_chars: int = 900) -> str:
    """
    Truncate very long posts so flan-t5 doesn't take forever.
    Keeps word boundary and adds ellipsis.
    """
    post = str(post or "")
    if len(post) <= max_chars:
        return post
    cut = post[:max_chars]
    cut = cut.rsplit(" ", 1)[0]
    return cut + "..."


def run_once(post, config, gen_model, device, timeout=120):
    """
    Run rag_generate.py once and return parsed dict or raise RuntimeError.
    - Truncates long posts.
    - Uses timeout if > 0.
    - Tries to salvage JSON from noisy stdout.
    """
    trimmed_post = truncate_post(post)

    cmd = [
        sys.executable,
        str(RAG),
        "-c",
        str(config),
        "--post",
        trimmed_post,
        "--gen_model",
        gen_model,
        "--device",
        device,
    ]

    try:
        if timeout and timeout > 0:
            p = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
        else:
            # timeout <= 0 means "no timeout"
            p = subprocess.run(cmd, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"TIMEOUT after {timeout}s for post: {trimmed_post[:80]!r}")

    stdout = (p.stdout or "").strip()
    stderr = (p.stderr or "").strip()

    if not stdout:
        raise RuntimeError(f"Empty stdout.\nSTDERR:\n{stderr}")

    # First try: direct JSON
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        # Try to salvage: last {...} block in stdout
        m = re.search(r"\{.*\}\s*$", stdout, flags=re.S | re.M)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        raise RuntimeError(
            f"Bad JSON from rag_generate.py.\n"
            f"STDOUT (truncated):\n{stdout[:500]}\n\nSTDERR:\n{stderr}"
        )


def normalize_label(x):
    """Force labels to simple strings; handle list/None."""
    if x is None:
        return None
    if isinstance(x, list):
        for item in x:
            if item and isinstance(item, str):
                return item.strip() or None
        return None
    if isinstance(x, str):
        x = x.strip()
        return x if x else None
    return str(x)


def main():
    ap = argparse.ArgumentParser(
        description="Generate predictions from gold posts using rag_generate.py (robust)."
    )
    ap.add_argument("--gold", required=True, help="gold jsonl with {post, intent, concern}")
    ap.add_argument("--out", required=True, help="output predictions jsonl")
    ap.add_argument("--config", required=True, help="RAG config YAML")
    ap.add_argument("--gen_model", default="google/flan-t5-large")
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    ap.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="per-call timeout in seconds; <=0 means no timeout",
    )
    ap.add_argument(
        "--sleep_between",
        type=float,
        default=0.0,
        help="seconds to sleep between calls",
    )
    ap.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="optional cap on number of gold rows to process (0 = all)",
    )
    ap.add_argument(
        "--start_row",
        type=int,
        default=1,
        help="1-based index of first gold row to process (useful for resuming)",
    )
    args = ap.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"[FATAL] Gold file not found: {gold_path}", file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = 0
    intent_vals = []
    concern_vals = []

    # Stream reading + streaming write so we never end up with an empty file if some rows succeed
    with open(gold_path, "r", encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, 1):
            if line_no < args.start_row:
                continue

            line = line.strip()
            if not line:
                continue

            if args.max_rows and (n_ok + n_fail) >= args.max_rows:
                break

            try:
                row = json.loads(line)
            except Exception as e:
                print(
                    f"[WARN] Skipping bad JSONL line {line_no}: {e}",
                    file=sys.stderr,
                )
                n_fail += 1
                continue

            post = row.get("post", "")
            if not post:
                print(
                    f"[WARN] Line {line_no} has no 'post'; skipping.",
                    file=sys.stderr,
                )
                n_fail += 1
                continue

            try:
                out = run_once(
                    post,
                    args.config,
                    args.gen_model,
                    args.device,
                    timeout=args.timeout,
                )
                err_msg = None
                n_ok += 1
            except Exception as e:
                # Don't crash – just record a stub prediction with error info
                err_msg = f"{type(e).__name__}: {e}"
                print(
                    f"[ERROR] Generation failed for line {line_no}: {err_msg}",
                    file=sys.stderr,
                )
                out = {}
                n_fail += 1

            pred_intent = normalize_label(out.get("predicted_intents"))
            pred_concern = normalize_label(out.get("predicted_concern"))

            # Record labels if present (for quick counts)
            if pred_intent:
                intent_vals.append(pred_intent)
            if pred_concern:
                concern_vals.append(pred_concern)

            pred = {
                "post": post,
                "predicted_intents": pred_intent,
                "predicted_concern": pred_concern,
                "reply": out.get("reply", ""),
                "citations": out.get("citations", []),
            }
            if err_msg:
                pred["error"] = err_msg

            fout.write(json.dumps(pred, ensure_ascii=False) + "\n")
            fout.flush()

            # Light progress logging
            if (n_ok + n_fail) % 25 == 0:
                print(
                    f"[PROGRESS] processed {n_ok + n_fail} rows "
                    f"(ok={n_ok}, fail={n_fail})",
                    file=sys.stderr,
                )

            if args.sleep_between > 0:
                time.sleep(args.sleep_between)

    # Final summary
    print(
        f"\nWrote {n_ok + n_fail} predictions to {out_path} (ok={n_ok}, fail={n_fail})"
    )
    print("Intent counts:", dict(Counter(intent_vals)))
    print("Concern counts:", dict(Counter(concern_vals)))


if __name__ == "__main__":
    main()