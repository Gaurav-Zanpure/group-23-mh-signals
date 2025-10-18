# test_transformers.py
import sys
import transformers
from transformers import TrainingArguments

print(f"--- Minimal Test ---")
print(f"Python Executable: {sys.executable}")
print(f"Transformers Version: {transformers.__version__}")
print("-" * 20)
print("Attempting to instantiate TrainingArguments...")

try:
    args = TrainingArguments(
        output_dir="./test_output",
        evaluation_strategy="epoch",  # The exact argument that fails
    )
    print("\n✅ SUCCESS: TrainingArguments instantiated correctly.")

except TypeError as e:
    print(f"\n❌ FAILURE: The TypeError occurred even in a minimal script.")
    print(f"   Error: {e}")

print("-" * 20)