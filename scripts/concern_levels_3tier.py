import sys, pandas as pd

in_path  = sys.argv[1] if len(sys.argv) > 1 else "this_tagged.csv"
out_path = sys.argv[2] if len(sys.argv) > 2 else "this_with_concern.csv"

df = pd.read_csv(in_path)

text_col = "Text" if "Text" in df.columns else df.columns[0]
tag_col  = "Tags" if "Tags" in df.columns else df.columns[-1]

ALL = {
    "critical risk","mental distress","maladaptive coping","positive coping",
    "seeking help","progress update","mood tracking","cause of distress"
}

def parse_tags(s):
    return [t.strip() for t in str(s).split(",") if t.strip()]

def concern_from_tags(tags_str: str):
    tl = {t.lower() for t in parse_tags(tags_str)}
    tl = {t for t in tl if t in ALL}

    if "critical risk" in tl:
        notes = []
        if "maladaptive coping" in tl: notes.append("with Maladaptive Coping")
        if "progress update" in tl:    notes.append("with Progress Update")
        if "seeking help" in tl:       notes.append("and Seeking Help")
        reason = "Critical Risk"
        return "High", reason, ("; ".join(notes) if notes else "")

    if tl & {"seeking help","maladaptive coping","progress update"}:
        reason = ", ".join(sorted(x.title() for x in tl & {"seeking help","maladaptive coping","progress update"}))
        notes  = ", ".join(sorted(x.title() for x in tl & {"mental distress","cause of distress"}))
        return "Medium", reason, notes
    if tl & {"mental distress","cause of distress"}:
        reason = ", ".join(sorted(x.title() for x in tl & {"mental distress","cause of distress"}))
        notes  = ", ".join(sorted(x.title() for x in tl & {"positive coping","mood tracking"}))
        return "Medium", reason, notes

    reason = ", ".join(sorted(x.title() for x in tl)) or "No higher-severity tags"
    return "Low", reason, ""

levels, reasons, notes = [], [], []
for s in df[tag_col].astype(str):
    L, R, N = concern_from_tags(s)
    levels.append(L); reasons.append(R); notes.append(N)

df.insert(len(df.columns), "Concern_Level", levels)

df.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
