import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from enrich_question_features import enrich_question_features
from model_features import FEATURE_COLUMNS, enrich_model_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "mmlu_pro_full.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "mmlu_pro_full_enriched.csv"

BASE_COLUMNS = ["question", "llm_model", "error"]
TRAINING_OUTPUT_COLUMNS = BASE_COLUMNS + FEATURE_COLUMNS + [
    "question_length_words",
    "question_length_chars",
    "question_complexity_score",
    "has_few_shot_examples",
    "prompt_contains_system_instructions",
    "question_category",
    "is_ambiguous",
    "contains_negation",
    "context_token_count",
]


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing {INPUT_PATH}. Run scripts/build_mmlu_pro_dataset.py first.")

    df = pd.read_csv(INPUT_PATH, usecols=["question", "llm_model", "error", "category"])
    enriched = enrich_model_features(df)
    enriched = enrich_question_features(enriched)
    enriched = enriched[TRAINING_OUTPUT_COLUMNS]
    enriched.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(enriched)} rows and {len(enriched.columns)} columns to {OUTPUT_PATH}")
    print(enriched["error"].value_counts().to_string())
    print()
    print(f"Models: {enriched['llm_model'].nunique()}")
    print()
    print(enriched.groupby("llm_model")[FEATURE_COLUMNS].first().head(3).to_string())
    print()
    print(enriched[BASE_COLUMNS + ["question_category", "context_window_tokens"]].head(2).to_string())


if __name__ == "__main__":
    main()
