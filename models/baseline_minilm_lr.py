import argparse
import json
import math
import random
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


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


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)


def load_yaml(p: str):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _normalize_tag(t: str) -> str | None:
    x = t.strip().lower()
    x = re.sub(r"\.$", "", x)
    x = x.replace("causes of distress", "cause of distress")
    x = x.replace("progress update.", "progress update")
    if x in CANONICAL:
        return CANONICAL[x]
    return None


def read_split_csv(p: Path):
    df = pd.read_csv(p)

    # FIXED: Accept Final_Tags instead of Tag
    if "Post" not in df.columns and "Text" in df.columns:
        df = df.rename(columns={"Text": "Post"})
    if "Post" not in df.columns:
        raise ValueError(f"'Post' column missing in {p}")
    if "Final_Tags" not in df.columns:
        raise ValueError(f"'Final_Tags' column missing in {p}")

    df = df.rename(columns={"Final_Tags": "Tag"})
    df["Post"] = df["Post"].fillna("").astype(str)

    def to_canonical_list(x):
        if isinstance(x, float) and math.isnan(x):
            raw = []
        else:
            s = str(x).strip()
            # ---- FIX: parse list-like strings ['Mental Distress', 'Mood Tracking'] ----
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]  # remove [ ]
                raw = [item.strip().strip("'").strip('"') for item in s.split(",")]
            else:
                raw = re.split(r"[;,]", s)

        norm = []
        seen = set()
        for r in raw:
            can = _normalize_tag(r)
            if can and can not in seen:
                norm.append(can)
                seen.add(can)

        if not norm:
            norm = ["Miscellaneous"]

        return norm

    df["TagsList"] = df["Tag"].apply(to_canonical_list)
    return df[["Post", "TagsList"]]


def encode_texts(embedder, texts, batch_size=128):
    out = []
    for i in range(0, len(texts), batch_size):
        out.append(
            embedder.encode(
                texts[i : i + batch_size],
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        )
    return np.vstack(out)


def evaluate(y_true, y_prob, threshold=0.5, label_names=None):
    y_pred = (y_prob >= threshold).astype(int)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    ap = []
    for j in range(y_true.shape[1]):
        try:
            ap.append(average_precision_score(y_true[:, j], y_prob[:, j]))
        except ValueError:
            ap.append(0.0)
    pr_auc_macro = float(np.mean(ap)) if ap else 0.0
    per_label_f1 = {}
    if label_names is not None:
        for j, n in enumerate(label_names):
            per_label_f1[n] = f1_score(y_true[:, j], y_pred[:, j], zero_division=0)
    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "pr_auc_macro": float(pr_auc_macro),
        "per_label_f1": per_label_f1,
    }


def prob_to_tags(prob_row, threshold, names):
    idx = np.where(prob_row >= threshold)[0].tolist()
    if not idx:
        idx = [int(np.argmax(prob_row))]
    return ", ".join([names[i] for i in idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = load_yaml(cfg["data"]["data_cfg"])

    set_seed(int(cfg.get("training", {}).get("seed", 42)))

    run_name = cfg["logging"]["run_name"]
    save_root = Path(cfg["logging"]["save_dir"])
    save_dir = save_root / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(save_dir)
    ensure_dir(save_dir / "tables")

    splits_dir = Path(data_cfg["paths"]["splits_dir"])
    train_df = read_split_csv(splits_dir / "train.csv")
    val_df = read_split_csv(splits_dir / "val.csv")
    test_df = read_split_csv(splits_dir / "test.csv")

    mlb = MultiLabelBinarizer(classes=sorted(CANON_KEYS))
    Y_train = mlb.fit_transform(train_df["TagsList"])
    Y_val = mlb.transform(val_df["TagsList"])
    Y_test = mlb.transform(test_df["TagsList"])
    label_names = list(mlb.classes_)

    embedder_name = cfg["model"]["embedder"]
    embedder = SentenceTransformer(embedder_name)

    X_train = encode_texts(embedder, train_df["Post"].tolist())
    X_val = encode_texts(embedder, val_df["Post"].tolist())
    X_test = encode_texts(embedder, test_df["Post"].tolist())

    lr = LogisticRegression(
        max_iter=int(cfg["training"].get("max_iter", 200)),
        C=float(cfg["training"].get("C", 1.0)),
        class_weight=cfg["training"].get("class_weight", "balanced"),
        n_jobs=int(cfg["training"].get("n_jobs", -1)),
        random_state=int(cfg["training"].get("seed", 42)),
        solver="liblinear",
    )
    clf = OneVsRestClassifier(lr, n_jobs=int(cfg["training"].get("n_jobs", -1)))

    t0 = time.time()
    clf.fit(X_train, Y_train)
    train_time = time.time() - t0

    P_val = clf.predict_proba(X_val)
    P_test = clf.predict_proba(X_test)
    thr = float(cfg["training"].get("threshold", 0.5))

    metrics_val = evaluate(Y_val, P_val, thr, label_names)
    metrics_test = evaluate(Y_test, P_test, thr, label_names)

    preds_val = pd.DataFrame(
        {
            "Post": val_df["Post"],
            "True": [", ".join(t) for t in val_df["TagsList"]],
            "Pred": [prob_to_tags(p, thr, label_names) for p in P_val],
        }
    )
    preds_test = pd.DataFrame(
        {
            "Post": test_df["Post"],
            "True": [", ".join(t) for t in test_df["TagsList"]],
            "Pred": [prob_to_tags(p, thr, label_names) for p in P_test],
        }
    )

    preds_val.to_csv(save_dir / "tables" / "val_predictions.csv", index=False)
    preds_test.to_csv(save_dir / "tables" / "test_predictions.csv", index=False)

    with open(save_dir / "metrics_val.json", "w") as f:
        json.dump(metrics_val, f, indent=2)
    with open(save_dir / "metrics_test.json", "w") as f:
        json.dump(metrics_test, f, indent=2)
    with open(save_dir / "label_names.json", "w") as f:
        json.dump(label_names, f, indent=2)
    with open(save_dir / "used_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    with open(save_dir / "data_config.yaml", "w") as f:
        yaml.safe_dump(data_cfg, f)

    print(f"[DONE] Saved run to: {save_dir}")
    print(f"VAL  -> {metrics_val}")
    print(f"TEST -> {metrics_test}")
    print(f"Train time (s): {train_time:.2f}")
    print(f"Labels ({len(label_names)}): {label_names}")


if __name__ == "__main__":
    main()
