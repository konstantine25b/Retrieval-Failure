import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "misprompt"
OUTPUT_PATH = PROJECT_ROOT / "data" / "misprompt_full.csv"

SPLIT_FILES = {
    "train": "train.json",
    "dev": "dev.json",
    "eval": "eval.json",
}


def normalize_label(label: str) -> str:
    if label == "error":
        return "error"
    if label in {"correct", "no_error"}:
        return "no_error"
    raise ValueError(f"Unknown label: {label}")


def load_split_rows() -> list[dict]:
    rows = []
    for split, filename in SPLIT_FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run scripts/download_misprompt.py first.")
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            rows.append(
                {
                    "id": row["id"],
                    "split": split,
                    "question": row["prompt"].strip(),
                    "llm_model": "gpt-4o",
                    "error": normalize_label(row["label"]),
                    "primary_category": row.get("primary-category"),
                    "secondary_category": row.get("secondary-category"),
                    "explanation": row.get("explanation"),
                    "gold_answer": row.get("gold-answer"),
                }
            )
    return rows


def main():
    rows = load_split_rows()
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df["error"].value_counts().to_string())
    print()
    print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
