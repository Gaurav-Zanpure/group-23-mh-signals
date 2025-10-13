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
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)


def load_yaml(p: str):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _norm_concern(x: str) -> str | None:
    if not isinstance(x, str):
        return None
    t = x.strip().lower()
    t = re.sub(r"[.\s]+$", "", t)
    if t in {"low", "medium", "high"}:
        return t
    if t in {"med", "mid"}:
        return "medium"
    return None


def read_split_csv(p: Path):
    df = pd.read_csv(p)
    if "Post" not in df.columns and "Text" in df.columns:
        df = df.rename(columns={"Text": "Post"})
    if "Post" not in df.columns:
        raise ValueError(f"'Post' column missing in {p}")
    if "Concern_Level" not in df.columns:
        raise ValueError(f"'Concern_Level' column missing in {p}")
    df["Post"] = df["Post"].fillna("").astype(str)
    df["Concern_Level"] = df["Concern_Level"].apply(_norm_concern)
    df = df.dropna(subset=["Concern_Level"]).reset_index(drop=True)
    return df[["Post", "Concern_Level"]]


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


def evaluate(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_label_f1 = {}
    for i, name in enumerate(labels):
        per_label_f1[name] = f1_score(
            (y_true == i).astype(int),
            (y_pred == i).astype(int),
            zero_division=0,
        )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist()
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_label_f1": per_label_f1,
        "confusion_matrix": cm,
        "label_order": labels,
    }


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

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["Concern_Level"])
    y_val = le.transform(val_df["Concern_Level"])
    y_test = le.transform(test_df["Concern_Level"])
    label_names = list(le.classes_)

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
        solver=cfg["training"].get("solver", "lbfgs"),
        multi_class=cfg["training"].get("multi_class", "auto"),
    )

    t0 = time.time()
    lr.fit(X_train, y_train)
    train_time = time.time() - t0

    y_val_pred = lr.predict(X_val)
    y_test_pred = lr.predict(X_test)

    metrics_val = evaluate(y_val, y_val_pred, label_names)
    metrics_test = evaluate(y_test, y_test_pred, label_names)

    pd.DataFrame(
        {
            "Post": val_df["Post"],
            "True": [label_names[i] for i in y_val],
            "Pred": [label_names[i] for i in y_val_pred],
        }
    ).to_csv(save_dir / "tables" / "val_predictions.csv", index=False)

    pd.DataFrame(
        {
            "Post": test_df["Post"],
            "True": [label_names[i] for i in y_test],
            "Pred": [label_names[i] for i in y_test_pred],
        }
    ).to_csv(save_dir / "tables" / "test_predictions.csv", index=False)

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
