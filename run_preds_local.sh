#!/bin/bash

echo "===== Local RAG Prediction Run ====="

# Move into project directory (optional)
cd "$(dirname "$0")"
#cd /Users/pragya/Documents/GitHub/group-23-mh-signals 2>/dev/null || true
cd /Users/vanshikawadhwa/Documents/NLP/group-23-mh-signals 2>/dev/null || true
echo "Current directory: $(pwd)"

# --- Create + Activate Virtual Environment ---
echo "===== Setting up virtual environment ====="

if [ ! -d ".venv11" ]; then
    echo "No venv found — creating a new one..."
    python3 -m venv .venv311
    source .venv311/bin/activate

    echo "Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Existing venv found — activating it..."
    source .venv311/bin/activate

    echo "Ensuring requirements are installed..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

echo "Venv active and requirements installed."

echo "===== Running Prediction Script ====="

./.venv311/bin/python scripts/make_preds_from_gold.py \
  --gold data/splits/test_gold.jsonl \
  --out data/splits/test_pred.jsonl \
  --config configs/data.yaml \
  --gen_model google/flan-t5-large \
  --device mps \
  --timeout 300 \
  --sleep_between 0.01

echo "===== Done ====="
