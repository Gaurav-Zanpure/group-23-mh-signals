import argparse
import json
import torch
from torch.nn import BCEWithLogitsLoss
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    # DataCollatorWithPadding,
    EvalPrediction,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
# from .helper import (
#     CANON_KEYS,
#     set_seed,
#     load_yaml,
#     ensure_dir,
#     read_split_csv,
#     prob_to_tags,
# )
from .helper_distilroberta import (
    CANON_KEYS,
    set_seed,
    load_yaml,
    ensure_dir,
    read_split_csv,
    prob_to_tags,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Multi-label BCE loss with class weights
        loss_fct = BCEWithLogitsLoss(pos_weight=self.class_weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

def main(args):
    # Load configuration
    cfg = load_yaml(args.config)
    data_cfg = load_yaml(cfg["data"]["data_cfg"])
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    lora_cfg = cfg.get("lora", {})
    # log_cfg = cfg.get("logging", {})

    set_seed(train_cfg["seed"])

    # Timestamp for unique run directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = cfg["logging"]["run_name"]
    save_root = Path(cfg["logging"]["save_dir"])
    save_dir = save_root / f"{run_name}_{timestamp}"
    ensure_dir(save_dir)
    ensure_dir(save_dir / "tables")

    # # Build output directories
    # base_output_dir = f"./results/distilroberta_lora_{timestamp}"
    # base_table_dir = f"./tables/distilroberta_{timestamp}"
    # ensure_dir(base_output_dir)
    # ensure_dir(base_table_dir)
    # ensure_dir(log_cfg.get("save_dir", "results/runs"))

    logger.info(f"Run started for DistilRoBERTa + LoRA | Timestamp: {timestamp}")

    # Load dataset
    logger.info("Loading dataset...")
    # train_path = data_cfg["train_path"]
    # test_path = data_cfg["test_path"]
    # dataset = load_dataset("csv", data_files={"train": train_path, "test": test_path})
    splits_dir = Path(data_cfg["paths"]["splits_dir"])
    train_df = read_split_csv(splits_dir / "train.csv")
    val_df = read_split_csv(splits_dir / "val.csv")
    test_df = read_split_csv(splits_dir / "test.csv")

    mlb = MultiLabelBinarizer(classes=sorted(CANON_KEYS))
    Y_train = mlb.fit_transform(train_df["TagsList"])
    Y_val = mlb.transform(val_df["TagsList"])
    Y_test = mlb.transform(test_df["TagsList"])
    label_names = list(mlb.classes_)

    class_counts = Y_train.sum(axis=0)  # number of positive examples per class
    class_weights = 1.0 / (class_counts + 1e-5)  # inverse frequency
    class_weights = class_weights / class_weights.sum() * len(class_counts)  # normalize
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Load tokenizer
    logger.info("Loading DistilRoBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])

    def create_dataset(df, y):
        ds = Dataset.from_pandas(df)
        ds = ds.add_column("labels", [row.astype(np.float32) for row in y])
        return ds

    def preprocess_function(examples):
        return tokenizer(
            examples["Post"], truncation=True, padding="max_length", max_length=model_cfg.get("max_length", 512)
        )

    # tokenized_datasets = dataset.map(preprocess_function, batched=True)
    train_ds = create_dataset(train_df, Y_train).map(preprocess_function, batched=True)
    val_ds = create_dataset(val_df, Y_val).map(preprocess_function, batched=True)
    test_ds = create_dataset(test_df, Y_test).map(preprocess_function, batched=True)

    # Model setup
    num_labels = model_cfg["num_labels"]
    logger.info(f"Initializing model: {model_cfg['name']} with {num_labels} labels")
    model = AutoModelForSequenceClassification.from_pretrained(model_cfg["name"], num_labels=len(label_names), problem_type="multi_label_classification")

    # LoRA setup
    logger.info("Applying LoRA configuration...")
    # lora_config = LoraConfig(
    #     r=lora_cfg.get("r", 4),
    #     lora_alpha=lora_cfg.get("lora_alpha", 8),
    #     lora_dropout=lora_cfg.get("lora_dropout", 0.1),
    #     target_modules=lora_cfg.get("target_modules", ["q_lin", "v_lin"]),
    #     bias="none",
    #     task_type="SEQ_CLS",
    # )
    # for name, module in model.named_modules():
    #     print("#"*75)
    #     print(name)
    
    lora_config = LoraConfig(**lora_cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("LoRA successfully integrated into DistilRoBERTa.")

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
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate= float(train_cfg["learning_rate"]),
        per_device_train_batch_size=train_cfg["train_batch_size"],
        per_device_eval_batch_size=train_cfg["eval_batch_size"],
        num_train_epochs=train_cfg["epochs"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc_macro",
        # logging_dir=f"{train_cfg["output_dir"]}/logs",
        # logging_strategy="steps",
        # logging_steps=50,
        save_total_limit=2,
        seed=train_cfg["seed"],
        report_to="none",
        greater_is_better=True,
        push_to_hub=False,
    )

    # Metric and evaluation
    # metric = load_metric("accuracy")

    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     predictions = torch.argmax(torch.tensor(logits), dim=-1)
    #     return metric.compute(predictions=predictions, references=labels)

    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        # data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )

    # Train
    logger.info("Starting training for DistilRoBERTa + LoRA...")
    t0 = time.time()
    trainer.train()
    time_elapsed = time.time() - t0
    logger.info(f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # Evaluate  
    logger.info("Evaluating model...")
    # metrics = trainer.evaluate()
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

    # --- 7. Find Optimal Threshold and Re-evaluate ---
    logger.info("Finding optimal threshold on validation set...")
    # print("\n Finding optimal threshold on validation set...")
    best_threshold = 0.0
    best_f1 = 0.0
    
    # Iterate over a range of potential thresholds
    for threshold in np.arange(0.05, 0.95, 0.01):
        preds = (P_val > threshold).astype(int)
        # Calculate the F1 score
        f1 = f1_score(Y_val, preds, average="macro", zero_division=0)
        # If this F1 is the best so far, save the threshold and F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Optimal threshold found: {best_threshold:.2f}")
    print(f"Best Macro F1 on Val set at this threshold: {best_f1:.4f}")

    # Update prediction dataframes with the optimized predictions
    preds_val_df["Pred"] = [prob_to_tags(p, best_threshold, label_names) for p in P_val]
    preds_test_df["Pred"] = [prob_to_tags(p, best_threshold, label_names) for p in P_test]

    # Now, use the BEST threshold to get the TRUE test metrics
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

    # The original metrics (with F1=0.0) are from val_preds.metrics
    with open(save_dir / "metrics_val_original.json", "w") as f: json.dump(val_preds.metrics, f, indent=2)
    # Save our new, meaningful metrics
    with open(save_dir / "metrics_test_optimized.json", "w") as f: json.dump(optimized_test_metrics, f, indent=2)
    with open(save_dir / "label_names.json", "w") as f: json.dump(label_names, f, indent=2)
    with open(save_dir / "used_config.yaml", "w") as f: yaml.safe_dump(cfg, f)
    with open(save_dir / "data_config.yaml", "w") as f: yaml.safe_dump(data_cfg, f)

    print(f"\n[DONE] Saved run to: {save_dir}")
    print(f"Optimized TEST Metrics (at threshold={best_threshold:.2f}) -> {json.dumps(optimized_test_metrics, indent=2)}")
    print(f"Train time (s): {time_elapsed:.2f}")

    # # Save model and metrics
    # logger.info("Saving model and metrics...")
    # trainer.save_model(base_output_dir)

    # import json

    # metrics_path = os.path.join(base_table_dir, "metrics.json")
    # with open(metrics_path, "w") as f:
    #     json.dump(metrics, f, indent=4)

    logger.info(f"Training complete. Metrics saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilRoBERTa with LoRA using YAML configuration.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="configs/distilroberta_lora.yaml"
    )
    args = parser.parse_args()
    main(args)
