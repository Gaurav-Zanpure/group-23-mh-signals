import pandas as pd
import argparse
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
from collections import Counter
import warnings
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score
import numpy as np


# --- Configuration ---
INPUT_FILE = "data/raw/mh_signal_data_w-intent.csv"
OUTPUT_DATASET_FILE = "data/full_dataset_tagged.csv"
OUTPUT_EVAL_FILE = "data/evaluation_report.csv"
CONFIDENCE_THRESHOLD = 0.5
BATCH_SIZE = 16

# --- Labels ---
ALL_LABELS = [
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

# We will only ask the model to predict the 8 semantic tags. "Miscellaneous" will be a fallback.
SEMANTIC_LABELS = [l for l in ALL_LABELS if l != "Miscellaneous"]
MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


def parse_human_tags(tag_string: str) -> list:
    """
    Converts a comma-separated string of tags into a clean list.
    """
    if not isinstance(tag_string, str) or not tag_string.strip():
        return []
    # Remove quotes, split by comma, strip whitespace
    tags = [tag.strip() for tag in tag_string.strip('"').split(',')]
    # Filter out any empty strings that might result
    return [tag for tag in tags if tag]


def run_classifier(texts: list, classifier, batch_size: int, threshold: float) -> list:
    """Helper function to run the pipeline and process results."""
    print(f"\nProcessing {len(texts)} posts...")
    all_tags = []
    
    for output in tqdm(
        classifier(
            texts,
            candidate_labels=SEMANTIC_LABELS,
            multi_label=True,
            batch_size=batch_size
        ),
        total=len(texts),
        desc="Classifying posts"
    ):
        tags = []
        for label, score in zip(output['labels'], output['scores']):
            if score > threshold:
                tags.append(label)
    
        all_tags.append(tags)
        
    return all_tags


def main(input_path, output_dataset, output_evaluation, threshold, batch_size):

    # --- Model Pipeline ---
    print("Setting up model pipeline...")
    warnings.filterwarnings("ignore", ".*Using a pipeline without specifying a model name.*")
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")
    if device == -1:
        print("WARNING: No GPU detected. This will be very slow.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=device)
    
    def truncate_text(text):
        tokens = tokenizer(
            text, truncation=True, max_length=tokenizer.model_max_length - 2
        )
        return tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)

    # --- Preparing Data ---
    print(f"\nReading: {input_path}")
    df = pd.read_csv(input_path)
    
    # Rename 'Post' to 'Text' if it exists, for consistency
    if 'Post' in df.columns and 'Text' not in df.columns:
        df = df.rename(columns={'Post': 'Text'})
        
    df["Text"] = df["Text"].fillna("").astype(str)
    df["Tag"] = df["Tag"].fillna("").astype(str)
    
    # Truncating text for the model
    df["Truncated_Text"] = df["Text"].apply(truncate_text)

    df_labeled = df[df['Tag'] != ""].copy()
    df_unlabeled = df[df['Tag'] == ""].copy()
    
    print(f"Found {len(df_labeled)} manually labeled posts.")
    print(f"Found {len(df_unlabeled)} unlabeled posts to tag.")

    final_labeled_rows = []
    final_unlabeled_rows = []

    # --- Processing Labeled Set for Evaluation ---
    if not df_labeled.empty:
        print("\n--- Evaluating Model on Manually tagged Set ---")
        df_labeled['Human_Tags'] = df_labeled['Tag'].apply(parse_human_tags)
        # Handling empty lists -> assign 'Miscellaneous'
        df_labeled['Human_Tags'] = df_labeled['Human_Tags'].apply(
            lambda tags: tags if tags else ['Miscellaneous']
        )
        
        texts_to_eval = df_labeled['Truncated_Text'].tolist()
        model_tags_eval = run_classifier(texts_to_eval, classifier, batch_size, threshold)
        df_labeled['Model_Tags'] = model_tags_eval
        # Assign 'Miscellaneous'
        df_labeled['Model_Tags'] = df_labeled['Model_Tags'].apply(
            lambda tags: tags if tags else ['Miscellaneous']
        )

        # Generating Evaluation Report
        print("\n--- Evaluation Report ---")
        # Use MultiLabelBinarizer to convert tag lists into a binary matrix
        mlb = MultiLabelBinarizer(classes=ALL_LABELS)
        
        y_true = mlb.fit_transform(df_labeled['Human_Tags'])
        y_pred = mlb.transform(df_labeled['Model_Tags'])

        report = classification_report(
            y_true, 
            y_pred, 
            target_names=mlb.classes_,
            zero_division=0
        )
        print(report)
        
        eval_cols = ['Text', 'Human_Tags', 'Model_Tags']
        df_labeled[eval_cols].to_csv(output_evaluation, index=False, encoding="utf-8")
        print(f"Saved side-by-side evaluation to: {output_evaluation}")

        # Prepare final dataset
        df_labeled['Final_Tags'] = df_labeled['Human_Tags'] # Use human tags as final
        df_labeled['Tag_Source'] = 'Human_Gold'
        final_labeled_rows = df_labeled
        
    # --- Process Unlabeled Set for Tagging ---
    if not df_unlabeled.empty:
        print("\n--- Tagging Unlabeled Set ---")
        texts_to_tag = df_unlabeled['Truncated_Text'].tolist()
        model_tags_new = run_classifier(texts_to_tag, classifier, batch_size, threshold)
        
        df_unlabeled['Model_Tags'] = model_tags_new
        # Assign 'Miscellaneous'
        df_unlabeled['Final_Tags'] = df_unlabeled['Model_Tags'].apply(
            lambda tags: tags if tags else ['Miscellaneous']
        )
        df_unlabeled['Tag_Source'] = 'Model'
        final_unlabeled_rows = df_unlabeled

    # --- Saving Final Dataset ---
    if final_labeled_rows is not None and not final_labeled_rows.empty:
        df_final = pd.concat([final_labeled_rows, final_unlabeled_rows])
    else:
        df_final = final_unlabeled_rows
        
    final_cols = ['Text', 'Tag', 'Final_Tags', 'Tag_Source']
    other_cols = [c for c in df.columns if c not in final_cols and c not in ['Truncated_Text']]
    
    df_final = df_final[other_cols + final_cols]

    df_final.to_csv(output_dataset, index=False, encoding="utf-8")
    print(f"\nSaved full {len(df_final)}-post tagged dataset to: {output_dataset}\n")
    print("--- Process Complete ---")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Evaluate and tag mental health dataset using Zero-Shot NLP."
    # )
    # parser.add_argument(
    #     "--input_path", 
    #     type=str, 
    #     required=True, 
    #     help="Path to input CSV (must have a 'Text' or 'Post' column)."
    # )
    # parser.add_argument(
    #     "--output_dataset",
    #     type=str,
    #     required=True,
    #     help="Path to save the FULL, combined (6000 post) tagged dataset."
    # )
    # parser.add_argument(
    #     "--output_evaluation",
    #     type=str,
    #     required=True,
    #     help="Path to save the side-by-side (500 post) evaluation report."
    # )
    # parser.add_argument(
    #     "--threshold",
    #     type=float,
    #     default=0.5,
    #     help="Confidence threshold (0.0 to 1.0) to assign a tag. Default: 0.5"
    # )
    # parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=16,
    #     help="Batch size for the model. Adjust based on GPU VRAM. Default: 16"
    # )
    
    # args = parser.parse_args()
    # main(
    #     args.input_path, 
    #     args.output_dataset, 
    #     args.output_evaluation, 
    #     args.threshold, 
    #     args.batch_size
    # )
    print("Starting the tagging and evaluation process...")
    main(
        INPUT_FILE,
        OUTPUT_DATASET_FILE,
        OUTPUT_EVAL_FILE,
        CONFIDENCE_THRESHOLD,
        BATCH_SIZE
        )
    print("All tasks finished.")
    