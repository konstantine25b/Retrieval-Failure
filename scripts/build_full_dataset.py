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


def extract_question(row: dict) -> str:
    text = row["input"]
    task = row["metadata"]["task_name"]

    if task == "answerability_classification" and "Question:\n" in text:
        return text.split("Question:\n", 1)[1].strip()

    if task == "finegrained_fact_verification":
        for line in text.split("\n"):
            if line.startswith("Claim"):
                return line.split(":", 1)[1].strip()

    if task == "math_problem_generation" and "Specific Requirements:\n" in text:
        return text.split("Specific Requirements:\n", 1)[1].strip()

    return text.strip()


def load_rows() -> list[dict]:
    rows = []
    for task_dir in TASK_DIRS:
        for path in sorted((DATA_DIR / task_dir).glob("*.jsonl")):
            with path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    rows.append(
                        {
                            "question": extract_question(row),
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
