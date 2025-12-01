# MH-SIGNALS: Mental Health Signal Detector

**MH-SIGNALS** is a modular framework for detecting mental-health signals such as **Intent** and **Concern Level** in online support-group posts.  
It benchmarks multiple modeling strategies — **MiniLM + Logistic Regression**, **DistilRoBERTa-LoRA**, and **RoBERTa-LoRA** — within a unified YAML-driven pipeline.

---

## Repository Structure

```
configs/                  # YAML configs (data.yaml + per-model configs)
data/
  raw/                    # raw CSV files
  raw/kb                  # raw CSV for KB after adding intent + concern tags
  processed/              # cleaned + tagged datasets
  processed/kb            # KB embeddings, metadata, snippets and faiss index
  splits/                 # train/val/test CSVs
  splits/test_gold.jsonl  # rag validation inputs
  splits/test_pred.jsonl  # rag validation predictions
models/                   # training scripts per model
results/
  runs/                   # checkpoints, logs, predictions
  tables/                 # summary CSVs
scripts/                  # tagging + summaries + kb creation + generation + validation
utils/                    # helper functions
logs/                     # Logs for general and high risk inputs to RAG
```

---

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or using **conda**:

```bash
conda create --name mh_signals python=3.10
conda activate mh_signals
pip install -r requirements.txt
```

---

## Configuration

All dataset and path settings are centralized in:

```
configs/data.yaml
```

If you move or rename files, update paths only in this file.

Example (already set):

```yaml
raw_data_path: data/raw/mental_health_signal_detector_data.csv
tagged_data_path: data/processed/tagged_posts.csv
splits_dir: data/splits
processed_dir: data/processed
results_dir: results
```

---

## Data and Tagging

Tagging has been completed.

* Tagged dataset: `data/processed/tagged_posts.csv`
* Tag summary: `data/processed/tags_summary.csv`
* Tag rules: `scripts/tag_rules_assign.py`

If you update raw data, re-run:

```bash
python scripts/tag_rules_assign.py
python scripts/tag_summary.py
```

---

## Pipeline Overview

1. **Input:** Raw CSV → cleaned and tagged posts  
2. **Splitting:** Fixed train/val/test CSVs in `data/splits/`  
3. **Training:** Model configuration read from YAML  
4. **Evaluation:** Metrics and predictions generated automatically  
5. **Results:** Saved to `results/runs/<run_id>/` and summarized in `results/tables/`  

All paths, splits, and outputs are handled through YAML — no manual edits required.

---

## Running Models

Each model has a YAML configuration under `configs/`.

### MiniLM + Logistic Regression (Baseline)

```bash
python models/baseline_minilm_lr.py --config configs/baseline_minilm.yaml
```

### DistilRoBERTa + LoRA (Medium)

```bash
python models/distilroberta_lora.py --config configs/distilroberta_lora.yaml
```

### RoBERTa-base + LoRA (Strong)

```bash
# python models/roberta_lora.py --config configs/roberta_lora.yaml
python -m models.roberta_lora --config configs/roberta_lora.yaml
```

Each script automatically:

* Loads paths from `configs/data.yaml`
* Uses the same train/val/test split
* Saves checkpoints, logs, and metrics

---

## Adding a New Model

To integrate a new model:

1. Create a new YAML file under `configs/`
   (refer to existing ones such as `baseline_minilm.yaml`)
2. Define:

   * `data`: path to `configs/data.yaml`
   * `training`: epochs, batch size, learning rate, etc.
   * `model`: name or Hugging Face ID
   * `logging`: run name and output folder
3. Run training:

```bash
python models/<trainer_script>.py --config configs/<your_model>.yaml
```

Outputs are stored under:

```
results/runs/<your_model>/
```

---

## Outputs

| Type               | Location                                |
| ------------------ | --------------------------------------- |
| Checkpoints & Logs | `results/runs/<run_id>/`                |
| Predictions        | `results/runs/<run_id>/predictions.csv` |
| Summary Metrics    | `results/tables/`                       |

Each run is self-contained, ensuring reproducibility.

---

## Notes

* All file paths are centralized in `configs/data.yaml`.
* Each model’s YAML fully defines its setup.
* Data splits are consistent across models.
* Results automatically save under `results/runs/`.

---

## RAG (Retrieval-Augmented Generation)

### Overview

The RAG system retrieves relevant counselor responses from a knowledge base and generates empathetic, grounded replies to mental health posts. It includes:

- **Knowledge Base Creation:** Embedding and indexing counselor responses
- **Generation:** Flan-T5-based reply generation with safety checks
- **Validation:** Quality evaluation of generated responses

### Prerequisites

1. **Build the Knowledge Base** (one-time setup):

```bash
python scripts/kb_build.py --config configs/data.yaml
```

This creates:
- `data/processed/kb/kb_snippets.jsonl` (chunked responses)
- `data/processed/kb/kb_embeddings.npy` (embeddings)
- `data/processed/kb/kb.faiss` (FAISS index)
- `data/processed/kb/kb_meta.jsonl` (metadata)

### Generating Responses

#### Single Post Generation

```bash
python scripts/rag_generate.py \
  --config configs/data.yaml \
  --post "I can't focus before my exams and I'm panicking." \
  --concern High \
  --keep 3 \
  --gen_model google/flan-t5-xl \
  --device mps
```

**Parameters:**
- `--post`: User's mental health post (required)
- `--concern`: Predicted concern level (Low/Medium/High), optional
- `--intents`: Space-separated intent tags, optional
- `--keep`: Number of KB snippets to retrieve (default: 3)
- `--topk`: Number of candidates before filtering (default: 50)
- `--gen_model`: Generator model (default: `google/flan-t5-xl`)
- `--device`: Device for generation (`mps`, `cuda`, or `cpu`)
- `--max_new_tokens`: Max tokens to generate (default: 400)

**Output (JSON):**
```json
{
  "post": "...",
  "crisis_level": "none",
  "citations": [...],
  "reply": "...",
  "disclaimer": "..."
}
```

#### Full Test Set Generation

Generate predictions for the entire test set (or a large batch):

```bash
python scripts/make_preds_from_gold.py \
  --gold data/splits/test_gold.jsonl \
  --out data/splits/test_pred.jsonl \
  --config configs/data.yaml \
  --gen_model google/flan-t5-xl \
  --device mps \
  --timeout 180
```

**Key Parameters:**
- `--gold`: Input JSONL file with `{post, intent, concern}` format (required)
- `--out`: Output predictions JSONL file (required)
- `--config`: Path to data.yaml configuration (required)
- `--gen_model`: Generator model (default: `google/flan-t5-large`)
- `--device`: Device for generation (`cpu`, `mps`, or `cuda`)
- `--timeout`: Per-post timeout in seconds (default: 180, set to 0 for no timeout)
- `--sleep_between`: Delay between posts in seconds (default: 0.0)
- `--max_rows`: Limit number of posts to process (default: 0 = all)
- `--start_row`: Starting row for resumption (1-based index, default: 1)

**Features:**
- **Automatic truncation:** Long posts are truncated to 900 characters to prevent timeouts
- **Robust error handling:** Failed posts are logged but don't stop the batch
- **Streaming output:** Results are written incrementally (resumable if interrupted)
- **Progress tracking:** Prints status every 25 posts
- **Timeout protection:** Prevents individual posts from hanging indefinitely

**Output Format (JSONL):**
```json
{
  "post": "...",
  "predicted_intents": "Mental Distress",
  "predicted_concern": "High",
  "reply": "...",
  "citations": [...]
}
```

**Example: Resume from Row 100**
```bash
python scripts/make_preds_from_gold.py \
  --gold data/splits/test_gold.jsonl \
  --out data/splits/test_pred.jsonl \
  --config configs/data.yaml \
  --gen_model google/flan-t5-xl \
  --device mps \
  --start_row 100 \
  --max_rows 50
```

This processes rows 100-149, useful for resuming interrupted batches or parallel processing.

### Validating Response Quality

Evaluate the quality of generated responses using multiple metrics:

```bash
python scripts/validate_reply_quality.py \
  --pred data/splits/sample_rag_output.jsonl \
  --enc sentence-transformers/all-MiniLM-L6-v2
```

**Evaluation Metrics:**

1. **Relevance** (30%): Semantic similarity between post and reply
2. **Grounding** (45%): How well the reply uses KB snippets (semantic + lexical overlap)
3. **Safety** (20%): Absence of harmful content or dangerous advice
4. **Crisis Coverage** (5%): Appropriate crisis resources when needed

### Logs

All RAG interactions are logged for review:

- **General:** `logs/general_YYYYMMDD.jsonl`
- **High-risk:** `logs/high_risk_YYYYMMDD.jsonl`

Each log entry includes post hash, reply hash, crisis level, and timestamp for auditing.

---

**Team Members:**  
Gaurav Zanpure · Shreyansh Kabra · Pragya Dhawan · Suyash Roy · Vanshika Wadhwa · Punith Basavaraj
