# MH-SIGNALS: Mental Health Signal Detector

**MH-SIGNALS** is a modular framework for detecting mental-health signals such as **Intent** and **Concern Level** in online support-group posts.  
It benchmarks multiple modeling strategies — **MiniLM + Logistic Regression**, **DistilRoBERTa-LoRA**, and **RoBERTa-LoRA** — within a unified YAML-driven pipeline.

---

## Repository Structure

```
configs/              # YAML configs (data.yaml + per-model configs)
data/
  raw/                # raw CSV files
  processed/          # cleaned + tagged datasets
  splits/             # train/val/test CSVs
models/               # training scripts per model
results/
  runs/               # checkpoints, logs, predictions
  tables/             # summary CSVs
scripts/              # tagging + summaries
utils/                # helper functions
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
python models/roberta_lora.py --config configs/roberta_lora.yaml
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

**Team Members:**  
Gaurav Zanpure · Shreyansh Kabra · Pragya Dhawan · Suyash Roy · Vanshika Wadhwa · Punith Basavaraj
