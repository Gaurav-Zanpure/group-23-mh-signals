# scripts/create_splits.py
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

INPUT = "data/raw/final/mh_signal_data_w-concern-intent.csv"
OUT = Path("data/splits")

COL_POST = "Post"
COL_TAGS = "Tag"
COL_CONCERN = "Concern_Level" 

df = pd.read_csv(INPUT)

df = df[[COL_POST, COL_TAGS, COL_CONCERN]].copy()
df[COL_TAGS] = df[COL_TAGS].fillna("")
df = df.dropna(subset=[COL_POST, COL_CONCERN])

df[COL_CONCERN] = df[COL_CONCERN].astype(str).str.strip().str.lower()

train, temp = train_test_split(
    df, test_size=0.30, random_state=42, shuffle=True, stratify=df[COL_CONCERN]
)
val, test = train_test_split(
    temp, test_size=0.50, random_state=42, shuffle=True, stratify=temp[COL_CONCERN]
)

OUT.mkdir(parents=True, exist_ok=True)
train.to_csv(OUT / "train.csv", index=False)
val.to_csv(OUT / "val.csv", index=False)
test.to_csv(OUT / "test.csv", index=False)

for name, split in [("train", train), ("val", val), ("test", test)]:
    print(name, split[COL_CONCERN].value_counts(normalize=True).round(3).to_dict())

print(f"Splits created at {OUT}/")
