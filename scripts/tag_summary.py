import pandas as pd, yaml, os
from collections import Counter

CANON = [
    "Critical Risk",
    "Mental Distress",
    "Maladaptive Coping",
    "Positive Coping",
    "Seeking Help",
    "Progress Update",
    "Mood Tracking",
    "Cause of Distress",
    "Miscellaneous",
]

ALIAS = {
    "critical risk":"Critical Risk",
    "mental distress":"Mental Distress",
    "maladaptive coping":"Maladaptive Coping",
    "positive coping":"Positive Coping",
    "seeking help":"Seeking Help",
    "progress update":"Progress Update",
    "mood tracking":"Mood Tracking",
    "cause of distress":"Cause of Distress",
    "cause of distress.":"Cause of Distress",
    "misc":"Miscellaneous",
    "miscellaneous":"Miscellaneous",
}

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def norm(tag):
    t = " ".join(str(tag).strip().split()).lower()
    return ALIAS.get(t, None)

def summarize_multi(tagged_csv, output_summary_csv="data/processed/tags_summary.csv"):
    if not os.path.exists(tagged_csv):
        raise FileNotFoundError(f"File not found: {tagged_csv}")
    df = pd.read_csv(tagged_csv)
    if "Tag" not in df.columns:
        raise KeyError("Column 'Tag' not found in CSV.")

    counts = Counter()
    unknown = Counter()

    for s in df["Tag"].fillna(""):
        labs = [x for x in map(str.strip, str(s).split(",")) if x]
        seen = set()
        for raw in labs:
            t = norm(raw)
            if t is None:
                unknown[raw.strip()] += 1
                continue
            if t in CANON and t not in seen:
                counts[t] += 1
                seen.add(t)

    total = sum(counts.get(t, 0) for t in CANON)
    rows = []
    for t in CANON:
        c = counts.get(t, 0)
        p = round(100 * c / total, 2) if total else 0.0
        rows.append({"Tag": t, "Count": c, "Percent": p})

    summary_df = pd.DataFrame(rows).sort_values("Count", ascending=False)
    os.makedirs(os.path.dirname(output_summary_csv), exist_ok=True)
    summary_df.to_csv(output_summary_csv, index=False, encoding="utf-8")
    print(summary_df.to_string(index=False))

    if unknown:
        unk_df = pd.DataFrame(sorted(unknown.items(), key=lambda x: -x[1]), columns=["UnknownTag","Count"])
        unk_path = os.path.join(os.path.dirname(output_summary_csv), "unknown_tags.csv")
        unk_df.to_csv(unk_path, index=False, encoding="utf-8")
        print(f"\nUnknown tags saved to: {unk_path}")

    print(f"\nSaved to: {output_summary_csv}")
    return summary_df

if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    summarize_multi(cfg["paths"]["tagged_data"], "data/processed/tags_summary.csv")
