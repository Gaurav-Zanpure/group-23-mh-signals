import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
import yaml
from datasets import Dataset
from peft import get_peft_model, LoraConfig
from packaging import version
from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from .helper import (
    CANON_KEYS,
    set_seed,
    load_yaml,
    ensure_dir,
    read_split_csv,
    prob_to_tags,
    read_and_process_data,
)
from .focal_loss import FocalLoss

class WeightedTrainer(Trainer):
    # def __init__(self, class_weights=None, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.class_weights = class_weights.to(self.model.device)

    # def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    #     labels = inputs.pop("labels")
    #     outputs = model(**inputs)
    #     logits = outputs.logits

    #     # Multi-label BCE loss with class weights
    #     loss_fct = BCEWithLogitsLoss(pos_weight=self.class_weights)
    #     loss = loss_fct(logits, labels)

    #     return (loss, outputs) if return_outputs else loss
    def __init__(self, class_weights=None, gamma = 2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = FocalLoss(gamma=gamma,
                                  pos_weight=class_weights)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        self.loss_fct = self.loss_fct.to(self.model.device)
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the model config YAML.")
    args = parser.parse_args()

    # Load Configs and Set Up Environment
    cfg = load_yaml(args.config)
    data_cfg = load_yaml(cfg["data"]["data_cfg"])
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    lora_cfg = cfg.get("lora", {})

    set_seed(train_cfg.get("seed", 42))

    run_name = cfg["logging"]["run_name"]
    save_root = Path(cfg["logging"]["save_dir"])
    save_dir = save_root / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(save_dir)
    ensure_dir(save_dir / "tables")

    # Load and Prepare Data
    # splits_dir = Path(data_cfg["paths"]["splits_dir"])
    # train_df = read_split_csv(splits_dir / "train.csv")
    # val_df = read_split_csv(splits_dir / "val.csv")
    # test_df = read_split_csv(splits_dir / "test.csv")

    print("Loading and splitting data dynamically...", flush=True)
    raw_data_path = Path(data_cfg["paths"]["llm_tagged_dir"]) / "full_dataset_tagged_BART.csv"
    all_df = read_and_process_data(raw_data_path)
    test_size = data_cfg["split"]["test_size"]
    val_size = data_cfg["split"]["val_size_from_train"]
    seed = train_cfg.get("seed", 42)
    train_val_df, test_df = train_test_split(
        all_df,
        test_size=test_size,
        random_state=seed,
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=seed,
    )
    print(f"Data split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test.")

    mlb = MultiLabelBinarizer(classes=sorted(CANON_KEYS))
    Y_train = mlb.fit_transform(train_df["TagsList"])
    Y_val = mlb.transform(val_df["TagsList"])
    Y_test = mlb.transform(test_df["TagsList"])
    label_names = list(mlb.classes_)

    class_counts = Y_train.sum(axis=0)
    class_weights = 1.0 / (class_counts + 1e-5) 
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Tokenize Data for Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])

    def create_dataset(df, y):
        ds = Dataset.from_pandas(df)
        ds = ds.add_column("labels", [row.astype(np.float32) for row in y])
        return ds

    def tokenize(batch):
        return tokenizer(batch["Post"], 
                         padding="max_length", 
                         truncation=True,
                         max_length=model_cfg.get("max_length", 512),
                         )

    train_ds = create_dataset(train_df, Y_train).map(tokenize, batched=True)
    val_ds = create_dataset(val_df, Y_val).map(tokenize, batched=True)
    test_ds = create_dataset(test_df, Y_test).map(tokenize, batched=True)

    # Configure Model, LoRA, and Metrics
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name"],
        num_labels=len(label_names),
        problem_type="multi_label_classification",
    )

    peft_config = LoraConfig(**lora_cfg)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def compute_metrics(p: EvalPrediction):
        logits, labels = p.predictions, p.label_ids
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > train_cfg["threshold"]).astype(int)
        
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        f1_micro = f1_score(labels, preds, average="micro", zero_division=0)

        pr_auc_macro = average_precision_score(labels, probs, average="macro")
        
        return {"macro_f1": f1_macro, 
                "micro_f1": f1_micro, 
                "pr_auc_macro": pr_auc_macro
                }

    # Set Up and Run Trainer
    args = {
        "output_dir":train_cfg["output_dir"],
        "learning_rate":float(train_cfg["learning_rate"]),
        "per_device_train_batch_size":train_cfg["train_batch_size"],
        "per_device_eval_batch_size":train_cfg["eval_batch_size"],
        "num_train_epochs":train_cfg["epochs"],
        "weight_decay":train_cfg["weight_decay"],
        "eval_strategy":"epoch",
        "save_strategy":"epoch",
        "load_best_model_at_end":True,
        "metric_for_best_model":"pr_auc_macro",
        "push_to_hub":False,
        # Mixed
        "warmup_steps": 500,
        "greater_is_better": True,
        "lr_scheduler_type": "cosine",
        "gradient_accumulation_steps": 4,
        "bf16":True,
    }
    if version.parse(transformers.__version__) >= version.parse("4.56.0"):
        args["eval_strategy"] = "epoch"
    else:
        args["evaluation_strategy"] = "epoch"
    
    gamma = train_cfg.get("gamma", 2.0)
    training_args = TrainingArguments(**args)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        gamma=gamma
    )
    print("Model and Trainer are set up. Starting training...")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print("Training completed.")

    # Evaluate and Save Results
    print("Evaluating model...")
    val_preds = trainer.predict(val_ds)
    test_preds = trainer.predict(test_ds)
    
    P_val = 1 / (1 + np.exp(-val_preds.predictions)) # Sigmoid
    P_test = 1 / (1 + np.exp(-test_preds.predictions)) # Sigmoid

    preds_val_df = pd.DataFrame({
        "Post": val_df["Post"],
        "True": [", ".join(t) for t in val_df["TagsList"]],
        "Pred": [prob_to_tags(p, train_cfg["threshold"], label_names) for p in P_val],
    })
    preds_test_df = pd.DataFrame({
        "Post": test_df["Post"],
        "True": [", ".join(t) for t in test_df["TagsList"]],
        "Pred": [prob_to_tags(p, train_cfg["threshold"], label_names) for p in P_test],
    })

    # Find Optimal Threshold and Re-evaluate
    print("\n Finding optimal threshold on validation set...")
    best_threshold = 0.0
    best_f1 = 0.0
    
    # Iterate over a range of potential thresholds
    for threshold in np.arange(0.05, 0.95, 0.01):
        preds = (P_val > threshold).astype(int)
        f1 = f1_score(Y_val, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Optimal threshold found: {best_threshold:.2f}")
    print(f"Best Macro F1 on Val set at this threshold: {best_f1:.4f}")

    # Update prediction dataframes with the optimized predictions
    preds_val_df["Pred"] = [prob_to_tags(p, best_threshold, label_names) for p in P_val]
    preds_test_df["Pred"] = [prob_to_tags(p, best_threshold, label_names) for p in P_test]

    # BEST threshold to get the TRUE test metrics
    print(f"\nRe-evaluating TEST set with new threshold of {best_threshold:.2f}...")
    preds_test_optimized = (P_test > best_threshold).astype(int)
    f1_macro_test_optimized = f1_score(Y_test, preds_test_optimized, average="macro", zero_division=0)
    f1_micro_test_optimized = f1_score(Y_test, preds_test_optimized, average="micro", zero_division=0)

    optimized_test_metrics = {
        "test_macro_f1": f1_macro_test_optimized,
        "test_micro_f1": f1_micro_test_optimized,
    }

    preds_val_df.to_csv(save_dir / "tables" / "val_predictions.csv", index=False)
    preds_test_df.to_csv(save_dir / "tables" / "test_predictions.csv", index=False)

    with open(save_dir / "metrics_val_original.json", "w") as f: json.dump(val_preds.metrics, f, indent=2)
    # Save our metrics
    with open(save_dir / "metrics_test_optimized.json", "w") as f: json.dump(optimized_test_metrics, f, indent=2)
    with open(save_dir / "label_names.json", "w") as f: json.dump(label_names, f, indent=2)
    with open(save_dir / "used_config.yaml", "w") as f: yaml.safe_dump(cfg, f)
    with open(save_dir / "data_config.yaml", "w") as f: yaml.safe_dump(data_cfg, f)

    print(f"\n[DONE] Saved run to: {save_dir}")
    print(f"Optimized TEST Metrics (at threshold={best_threshold:.2f}) -> {json.dumps(optimized_test_metrics, indent=2)}")
    print(f"Train time (s): {train_time:.2f}")

if __name__ == "__main__":
    main()
    