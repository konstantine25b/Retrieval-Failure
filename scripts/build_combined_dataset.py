from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REALMISTAKE_PATH = PROJECT_ROOT / "data" / "realmistake_full.csv"
MISPROMPT_PATH = PROJECT_ROOT / "data" / "misprompt_full.csv"
MMLU_PRO_PATH = PROJECT_ROOT / "data" / "mmlu_pro_full.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "combined_full.csv"


def main():
    if not REALMISTAKE_PATH.exists():
        raise FileNotFoundError(f"Missing {REALMISTAKE_PATH}")
    if not MISPROMPT_PATH.exists():
        raise FileNotFoundError(f"Missing {MISPROMPT_PATH}. Run scripts/build_misprompt_dataset.py first.")

    realmistake = pd.read_csv(REALMISTAKE_PATH)
    realmistake["source"] = "realmistake"
    realmistake["split"] = pd.NA

    misprompt = pd.read_csv(MISPROMPT_PATH)
    misprompt["source"] = "misprompt"

    parts = [realmistake, misprompt]

    if MMLU_PRO_PATH.exists():
        mmlu_pro = pd.read_csv(MMLU_PRO_PATH)
        mmlu_pro["source"] = "mmlu_pro"
        mmlu_pro["split"] = pd.NA
        mmlu_pro["primary_category"] = mmlu_pro.pop("category")
        mmlu_pro["secondary_category"] = pd.NA
        mmlu_pro["explanation"] = pd.NA
        mmlu_pro = mmlu_pro.drop(columns=["model_answer"], errors="ignore")
        parts.append(mmlu_pro)
    else:
        print(f"Skipping MMLU-Pro (missing {MMLU_PRO_PATH})")

    combined = pd.concat(parts, ignore_index=True, sort=False)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(combined)} rows to {OUTPUT_PATH}")
    print(combined["source"].value_counts().to_string())
    print()
    print(combined["error"].value_counts().to_string())
    print()
    print(combined.groupby(["source", "error"]).size().to_string())


if __name__ == "__main__":
    main()
