from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from enrich_question_features import enrich_question_features
from enrich_realmistake_full import enrich_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "combined_full.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "combined_full_enriched.csv"


def main():
    df = pd.read_csv(INPUT_PATH)
    enriched = enrich_dataframe(df)
    enriched = enrich_question_features(enriched)
    enriched.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(enriched)} rows and {len(enriched.columns)} columns to {OUTPUT_PATH}")
    print(enriched.groupby(["source", "error"]).size().to_string())


if __name__ == "__main__":
    main()
