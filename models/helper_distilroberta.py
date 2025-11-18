# models/helper_distilroberta.py
import math
import random
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml

# --- Canonical Tag Mapping ---
CANONICAL = {
    "critical risk": "Critical Risk", 
    "mental distress": "Mental Distress",
    "maladaptive coping": "Maladaptive Coping", 
    "positive coping": "Positive Coping",
    "seeking help": "Seeking Help", 
    "progress update": "Progress Update",
    "mood tracking": "Mood Tracking", 
    "cause of distress": "Cause of Distress",
    "miscellaneous": "Miscellaneous",
}
CANON_KEYS = set(CANONICAL.values())


# --- Utilities ---
def set_seed(s: int):
    """Set all random seeds for reproducibility."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_yaml(p: str):
    """Load YAML configuration from a given path."""
    with open(p, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path):
    """Ensure a directory exists."""
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- Data Processing Helpers ---
def _normalize_tag(t: str) -> str | None:
    """Normalize tag text to canonical form."""
    x = t.strip().lower()
    x = re.sub(r"\.$", "", x)
    x = x.replace("causes of distress", "cause of distress")
    x = x.replace("progress update.", "progress update")
    if x in CANONICAL:
        return CANONICAL[x]
    return None


def read_split_csv(p: Path):
    """Read train/val/test CSVs and normalize tag lists."""
    df = pd.read_csv(p)
    
    if "Post" not in df.columns and "Text" in df.columns:
        df = df.rename(columns={"Text": "Post"})
    if "Post" not in df.columns:
        raise ValueError(f"'Post' column missing in {p}")
    if "Final_Tags" not in df.columns:
        raise ValueError(f"'Tag' column missing in {p}")

    df["Post"] = df["Post"].fillna("").astype(str)

    def to_canonical_list(x):
        if isinstance(x, float) and math.isnan(x):
            return ["Miscellaneous"]

        s = str(x).strip()

        # Remove [] brackets
        s = s.strip("[]")

        # Split only on commas separating list elements
        parts = [p.strip().strip("'").strip('"') for p in s.split(",") if p.strip()]

        norm, seen = [], set()
        for r in parts:
            can = _normalize_tag(r)
            if can and can not in seen:
                norm.append(can)
                seen.add(can)

        return norm if norm else ["Miscellaneous"]

    df["TagsList"] = df["Final_Tags"].apply(to_canonical_list)
    return df[["Post", "TagsList"]]


def prob_to_tags(prob_row, threshold, names):
    """Convert probability vector to tag string based on threshold."""
    idx = np.where(prob_row >= threshold)[0].tolist()
    if not idx:
        idx = [int(np.argmax(prob_row))]
    return ", ".join([names[i] for i in idx])

def _normalize_concern_level(x: str) -> str | None:
    if not isinstance(x, str):
        return None
    t = x.strip().lower()
    t = re.sub(r"[.\s]+$", "", t)
    if t in {"low", "medium", "high"}:
        return t
    if t in {"med", "mid"}:
        return "medium"
    return None

def read_concern_split_csv(p: Path):
    df = pd.read_csv(p)
    if "Post" not in df.columns and "Text" in df.columns:
        df = df.rename(columns={"Text": "Post"})
    if "Post" not in df.columns:
        raise ValueError(f"'Post' column missing in {p}")
    if "Concern_Level" not in df.columns:
        raise ValueError(f"'Concern_Level' column missing in {p}")
    df["Post"] = df["Post"].fillna("").astype(str)
    df["Concern_Level"] = df["Concern_Level"].apply(_normalize_concern_level)
    df = df.dropna(subset=["Concern_Level"]).reset_index(drop=True)
    return df[["Post", "Concern_Level"]]