from pathlib import Path

import pandas as pd

from model_features import FEATURE_COLUMNS, enrich_model_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "realmistake_full.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "realmistake_full_enriched.csv"


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return enrich_model_features(df)


def main():
    df = pd.read_csv(INPUT_PATH)
    enriched = enrich_dataframe(df)
    enriched.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(enriched)} rows and {len(enriched.columns)} columns to {OUTPUT_PATH}")
    print(enriched["llm_model"].value_counts().to_string())
    print()
    print(enriched[FEATURE_COLUMNS].drop_duplicates().to_string(index=False))


if __name__ == "__main__":
    main()
