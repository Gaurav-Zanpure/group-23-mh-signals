import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
import yaml
from datasets import Dataset
from peft import get_peft_model, LoraConfig
from packaging import version
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from .helper import (
    set_seed,
    load_yaml,
    ensure_dir,
    read_concern_split_csv
)

# --- Custom Trainer for Weighted Loss ---

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = class_weights.to(self.model.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Single-label Cross-Entropy loss with class weights
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# --- Main Training and Evaluation Logic ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the model config YAML.")
    args = parser.parse_args()

    # 1. Load Configs and Set Up Environment
    cfg = load_yaml(args.config)
    data_cfg = load_yaml(cfg["data"]["data_cfg"])
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    lora_cfg = cfg.get("lora", {})

    set_seed(train_cfg.get("seed", 42)) # Using imported helper

    run_name = cfg["logging"]["run_name"]
    save_root = Path(cfg["logging"]["save_dir"])
    save_dir = save_root / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(save_dir) # Using imported helper
    ensure_dir(save_dir / "tables") # Using imported helper

    # 2. Load and Prepare Data
    splits_dir = Path(data_cfg["paths"]["splits_dir"])
    # Using the imported task-specific CSV reader
    train_df = read_concern_split_csv(splits_dir / "train.csv")
    val_df = read_concern_split_csv(splits_dir / "val.csv")
    test_df = read_concern_split_csv(splits_dir / "test.csv")

    # Use the label map from data.yaml
    label_map = data_cfg["labels"]["concern_map"]
    # Get label names in order of their index (0, 1, 2)
    label_names = sorted(label_map, key=label_map.get)
    
    Y_train = train_df["Concern_Level"].map(label_map).values
    Y_val = val_df["Concern_Level"].map(label_map).values
    Y_test = test_df["Concern_Level"].map(label_map).values

    num_labels = len(label_names)
    print(f"Found {num_labels} labels: {label_names}")

    # Calculate class weights for single-label classification
    class_counts = np.bincount(Y_train, minlength=num_labels)
    print(f"Class counts (Train): {list(zip(label_names, class_counts))}")
    
    class_weights = 1.0 / (class_counts + 1e-5)  # inverse frequency
    class_weights = class_weights / class_weights.sum() * num_labels  # normalize
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Applying class weights: {list(zip(label_names, class_weights.numpy()))}")


    # 3. Tokenize Data for Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])

    def create_dataset(df, y):
        ds = Dataset.from_pandas(df)
        # For single-label, labels are just the integer indices
        ds = ds.add_column("labels", y)
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

    # 4. Configure Model, LoRA, and Metrics
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name"],
        num_labels=num_labels,
        problem_type="single_label_classification", # Changed
    )

    peft_config = LoraConfig(**lora_cfg)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def compute_metrics(p: EvalPrediction):
        logits, labels = p.predictions, p.label_ids
        preds = np.argmax(logits, axis=1) # Use argmax for single-label
        
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        acc = accuracy_score(labels, preds)
        
        return {
            "accuracy": acc,
            "macro_f1": f1_macro,
        }

    # 5. Set Up and Run Trainer
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
        "metric_for_best_model":"macro_f1", # Changed from pr_auc_macro
        "push_to_hub":False,
        "warmup_steps": 500,
        "greater_is_better": True,
        "lr_scheduler_type": "cosine",
        "gradient_accumulation_steps": 4,
        "bf16":True,
    }
    # Handle transformers version compatibility for eval_strategy
    if version.parse(transformers.__version__) >= version.parse("4.56.0"):
        args["eval_strategy"] = "epoch"
    else:
        args["evaluation_strategy"] = "epoch"
    
    training_args = TrainingArguments(**args)

    trainer = WeightedTrainer( # Using our modified trainer
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )
    print("Model and Trainer are set up. Starting training...")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print("Training completed.")

    # 6. Evaluate and Save Results
    print("Evaluating model...")
    val_preds = trainer.predict(val_ds)
    test_preds = trainer.predict(test_ds)
    
    # Get predicted class index by argmax
    preds_val_idx = np.argmax(val_preds.predictions, axis=1)
    preds_test_idx = np.argmax(test_preds.predictions, axis=1)

    # Map indices back to string labels
    preds_val_labels = [label_names[i] for i in preds_val_idx]
    preds_test_labels = [label_names[i] for i in preds_test_idx]
    true_val_labels = [label_names[i] for i in Y_val]
    true_test_labels = [label_names[i] for i in Y_test]

    preds_val_df = pd.DataFrame({
        "Post": val_df["Post"],
        "True": true_val_labels,
        "Pred": preds_val_labels,
    })
    preds_test_df = pd.DataFrame({
        "Post": test_df["Post"],
        "True": true_test_labels,
        "Pred": preds_test_labels,
    })

    # --- 7. Save Final Metrics and Configs ---
    
    # Get metrics from the trainer.predict() call
    val_metrics = val_preds.metrics
    test_metrics = test_preds.metrics
    
    # Add confusion matrix to test metrics
    cm_test = confusion_matrix(Y_test, preds_test_idx).tolist()
    test_metrics["confusion_matrix"] = cm_test
    test_metrics["label_order"] = label_names

    preds_val_df.to_csv(save_dir / "tables" / "val_predictions.csv", index=False)
    preds_test_df.to_csv(save_dir / "tables" / "test_predictions.csv", index=False)

    with open(save_dir / "metrics_val.json", "w") as f: json.dump(val_metrics, f, indent=2)
    with open(save_dir / "metrics_test.json", "w") as f: json.dump(test_metrics, f, indent=2)
    with open(save_dir / "label_names.json", "w") as f: json.dump(label_names, f, indent=2)
    with open(save_dir / "used_config.yaml", "w") as f: yaml.safe_dump(cfg, f)
    with open(save_dir / "data_config.yaml", "w") as f: yaml.safe_dump(data_cfg, f)

    print(f"\n[DONE] Saved run to: {save_dir}")
    print(f"Test Metrics -> {json.dumps(test_metrics, indent=2)}")
    print(f"Train time (s): {train_time:.2f}")

if __name__ == "__main__":
    main()
    