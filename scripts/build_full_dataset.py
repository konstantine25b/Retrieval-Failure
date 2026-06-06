import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "realmistake"
OUTPUT_PATH = PROJECT_ROOT / "data" / "realmistake_full.csv"

TASK_DIRS = [
    "math_word_problem_generation",
    "finegrained_fact_verification",
    "answerability_classification",
]


def load_rows() -> list[dict]:
    rows = []
    for task_dir in TASK_DIRS:
        for path in sorted((DATA_DIR / task_dir).glob("*.jsonl")):
            with path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    rows.append(
                        {
                            "question": row["input"].strip(),
                            "llm_model": row["metadata"]["llm_response_model"],
                            "error": row["error_label"],
                        }
                    )
    return rows


def main():
    rows = load_rows()
    df = pd.DataFrame(rows, columns=["question", "llm_model", "error"])
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df["error"].value_counts().to_string())
    print(df["llm_model"].value_counts().to_string())


if __name__ == "__main__":
    main()
