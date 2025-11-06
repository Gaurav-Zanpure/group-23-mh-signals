import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import warnings
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, precision_recall_curve


# --- Configuration ---
INPUT_FILE = "data/llm_taged/mh_signal_data_w-intent.csv"
OUTPUT_DATASET_FILE = "data/llm_taged/full_dataset_tagged.csv"
OUTPUT_EVAL_FILE = "data/llm_taged/evaluation_report.csv"
BATCH_SIZE = 16
MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
# MODEL_NAME = "facebook/bart-large-mnli"

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

SEMANTIC_LABELS = [
    ("A person is in critical risk of suicide or self-harm", "Critical Risk"),
    ("A person is expressing feelings of depression, anxiety, or general distress", "Mental Distress"),
    ("A person is describing a negative coping mechanism (e.g., substance abuse, self-isolation)", "Maladaptive Coping"),
    ("A person is describing a positive coping mechanism (e.g., exercise, journaling, meditation)", "Positive Coping"),
    ("A person is asking for help, advice, or resources", "Seeking Help"),
    ("A person is sharing an update on their treatment, medication, or therapy", "Progress Update"),
    ("A person is sharing a log of their current mood or feelings", "Mood Tracking"),
    ("A person is identifying a specific reason for their distress (e.g., job, relationship, school)", "Cause of Distress")
]
SEMANTIC_LABEL_NAMES = [label for desc, label in SEMANTIC_LABELS]

def parse_human_tags(tag_string: str) -> list:
    """
    Converts a comma or semicolon-separated string of tags into a clean, canonical list.
    """
    if not isinstance(tag_string, str) or not tag_string.strip():
        return []

    label_map = {
        **{label.lower(): label for label in ALL_LABELS},
        **{label: label for label in ALL_LABELS},
        'cause of distress': 'Cause of Distress',
        'causes of distress': 'Cause of Distress',
        'progress update': 'Progress Update',
        'progress update. cause of distress': 'Progress Update',
    }

    tags = re.split(r'[;,]', tag_string)
    clean_tags = set()
    for tag in tags:
        tag_clean = tag.strip().strip('"')
        canonical_tag = label_map.get(tag_clean.lower())
        if canonical_tag:
            clean_tags.add(canonical_tag)

    if not clean_tags:
        return ["Miscellaneous"]

    return list(clean_tags)


def get_model_scores(texts: list, classifier, batch_size: int) -> list:
    """
    Runs the pipeline and returns a list of dictionaries containing the raw scores for each semantic label.
    """
    print(f"\nGetting model scores for {len(texts)} posts...")
    all_scores = []

    label_descriptions = [desc for desc, label in SEMANTIC_LABELS]
    description_to_label_map = {desc: label for desc, label in SEMANTIC_LABELS}

    for output in tqdm(
        classifier(
            texts,
            candidate_labels=label_descriptions,
            multi_label=True,
            batch_size=batch_size
        ),
        total=len(texts),
        desc="Classifying posts"
    ):
        score_dict = {label: 0.0 for label in SEMANTIC_LABEL_NAMES}
        for model_label, score in zip(output['labels'], output['scores']):
            simple_label = description_to_label_map[model_label]
            score_dict[simple_label] = score

        all_scores.append(score_dict)

    return all_scores

def find_optimal_thresholds(y_true_bin, y_pred_scores_df):
    """
    Finds the optimal F1 threshold for each semantic label.
    """
    print("\n--- Finding Optimal Thresholds ---")
    optimal_thresholds = {}

    for label in SEMANTIC_LABEL_NAMES:
        # Get the binary true labels for this category
        y_true_col = y_true_bin[label]
        y_pred_scores_col = y_pred_scores_df[label]

        precisions, recalls, thresholds = precision_recall_curve(
            y_true_col, y_pred_scores_col
        )
        # Adding a small epsilon (1e-9) to avoid division by zero
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)

        # Find the threshold that gives the highest F1 score
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx]

        optimal_thresholds[label] = best_threshold
        print(f"Optimal threshold for {label:<20}: {best_threshold:.4f} (F1: {f1_scores[best_f1_idx]:.4f})")

    return optimal_thresholds

def apply_thresholds(y_pred_scores_df, optimal_thresholds):
    """
    Applies the dictionary of optimal thresholds to the score dataframe to get the final tag lists.
    """
    print("\nApplying optimal thresholds to get final tags...")
    all_tags = []
    # Convert dataframe to list of dictionaries for iteration
    scores_list = y_pred_scores_df.to_dict('records')

    for post_scores in scores_list:
        tags = []
        for label, score in post_scores.items():
            # Check if the score meets the specific threshold for that label
            if score >= optimal_thresholds[label]:
                tags.append(label)

        if not tags:
            tags = ["Miscellaneous"]

        all_tags.append(tags)

    return all_tags


def main(input_path, output_dataset, output_evaluation, batch_size):

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
    df = pd.read_csv(input_path)
    
    # Rename 'Post' to 'Text' if it exists, for consistency
    if 'Post' in df.columns and 'Text' not in df.columns:
        df = df.rename(columns={'Post': 'Text'})
        
    df["Text"] = df["Text"].fillna("").astype(str)
    df["Tag"] = df["Tag"].fillna("").astype(str)
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
        mlb = MultiLabelBinarizer(classes=ALL_LABELS)
        y_true = mlb.fit_transform(df_labeled['Human_Tags'])
        y_true_bin_df = pd.DataFrame(y_true, columns=mlb.classes_)
    
        # Get model raw scores  
        texts_to_eval = df_labeled['Truncated_Text'].tolist()
        model_scores_list = get_model_scores(texts_to_eval, classifier, batch_size)
        y_pred_scores_df = pd.DataFrame(model_scores_list)

        # optimal thresholds
        optimal_thresholds = find_optimal_thresholds(y_true_bin_df, y_pred_scores_df)

        model_tags_eval = apply_thresholds(y_pred_scores_df, optimal_thresholds)
        df_labeled['Model_Tags'] = model_tags_eval
        y_pred_optimal = mlb.transform(model_tags_eval)

        print("\n--- Final Report (with Optimal Thresholds) ---")
        report = classification_report(
            y_true,
            y_pred_optimal,
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
        
        model_scores_list_new = get_model_scores(texts_to_tag, classifier, batch_size)
        y_pred_scores_df_new = pd.DataFrame(model_scores_list_new)
        
        print("\nApplying optimal thresholds to unlabeled data...")
        model_tags_new = apply_thresholds(y_pred_scores_df_new, optimal_thresholds)

        df_unlabeled['Final_Tags'] = model_tags_new
        df_unlabeled['Tag_Source'] = 'Model_Optimal'
        final_unlabeled_rows = df_unlabeled

    print("\n--- Saving Final Dataset ---")
    if final_labeled_rows is not None and not final_labeled_rows.empty:
        df_final = pd.concat([final_labeled_rows, final_unlabeled_rows])
    else:
        df_final = final_unlabeled_rows

    final_cols = ['Text', 'Tag', 'Final_Tags', 'Tag_Source']
    other_cols = [c for c in df.columns if c not in final_cols and c not in ['Truncated_Text', 'Post', 'Model_Tags']]

    df_final = df_final[other_cols + final_cols]

    df_final.to_csv(output_dataset, index=False, encoding="utf-8")
    print(f"\nSaved full {len(df_final)}-post tagged dataset to: {output_dataset}\n")
    print("--- Process Complete ---")

print("Starting the optimal tagging and evaluation process...")
main(
    INPUT_FILE,
    OUTPUT_DATASET_FILE,
    OUTPUT_EVAL_FILE,
    BATCH_SIZE
)
print("All tasks finished.")
