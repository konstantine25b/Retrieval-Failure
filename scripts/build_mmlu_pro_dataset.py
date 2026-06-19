import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "mmlu_pro" / "eval_results"
OUTPUT_PATH = PROJECT_ROOT / "data" / "mmlu_pro_full.csv"

OPTION_LETTERS = "ABCDEFGHIJ"
MODEL_NAME_PATTERN = re.compile(r"^model_outputs_(.+?)_(?:\d+shots|\d+-shots)(?:\.json)?$")


def parse_model_name(path: Path) -> str:
    match = MODEL_NAME_PATTERN.match(path.stem)
    if match:
        return match.group(1)
    return path.stem.replace("model_outputs_", "")


def format_question(row: dict) -> str:
    parts = [row["question"].strip(), ""]
    for index, option in enumerate(row.get("options") or []):
        if option and str(option).strip().upper() != "N/A":
            parts.append(f"{OPTION_LETTERS[index]}. {option}")
    return "\n".join(parts).strip()


def is_correct(row: dict) -> bool:
    pred = row.get("pred")
    answer = row.get("answer")
    if pred is None or str(pred).strip() == "":
        return False
    return str(pred).strip().upper() == str(answer).strip().upper()


def load_model_rows(path: Path) -> list[dict]:
    model_name = parse_model_name(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in data:
        if not isinstance(item, dict) or "question" not in item:
            continue
        rows.append(
            {
                "question": format_question(item),
                "llm_model": model_name,
                "error": "no_error" if is_correct(item) else "error",
                "id": item.get("question_id"),
                "category": item.get("category"),
                "gold_answer": item.get("answer"),
                "model_answer": item.get("pred"),
            }
        )
    return rows


def main():
    result_files = sorted(EVAL_DIR.glob("model_outputs_*.json"))
    if not result_files:
        raise FileNotFoundError(
            f"No eval result JSON files in {EVAL_DIR}. Run scripts/download_mmlu_pro.py first."
        )

    all_rows: list[dict] = []
    for path in result_files:
        rows = load_model_rows(path)
        all_rows.extend(rows)
        correct = sum(1 for row in rows if row["error"] == "no_error")
        print(f"{path.name}: {len(rows)} rows, accuracy {correct / len(rows):.1%}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(df)} rows to {OUTPUT_PATH}")
    print(f"Models: {df['llm_model'].nunique()}")
    print(df["error"].value_counts().to_string())
    print()
    print(df.groupby(["llm_model", "error"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
