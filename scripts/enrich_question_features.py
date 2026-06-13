import re
from pathlib import Path

import pandas as pd
import textstat
import tiktoken

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "realmistake_full_enriched.csv"
FALLBACK_INPUT_PATH = PROJECT_ROOT / "data" / "realmistake_full.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "realmistake_full_enriched.csv"

FEW_SHOT_PATTERN = re.compile(r"(?i)example\s+\d+\s*:")
SYSTEM_INSTRUCTION_PATTERN = re.compile(
    r"(?i)(?:^|\n)\s*system\s*:|(?:^|\n)\s*instructions?\s*:"
)
NEGATION_PATTERN = re.compile(
    r"(?i)\b(?:not|never|no|neither|nor|without|n't|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|shouldn't|wouldn't|couldn't|mustn't|hardly|barely)\b"
)

QUESTION_FEATURE_COLUMNS = [
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

GPT4_ENCODING = tiktoken.get_encoding("cl100k_base")


def classify_question_category(question: str) -> str:
    text = question.lower()
    if re.search(r"(?i)\b(def|class|import|function|python|javascript|sql|algorithm|code)\b", text):
        return "Coding"
    if re.search(r"(?i)\b(story|poem|creative|fiction|narrative|write a song)\b", text):
        return "Creative"
    if "claim and evidence" in text or "wikipedia article" in text or "fact verification" in text:
        return "Fact Retrieval"
    if "math word problem" in text or "generate a math" in text:
        return "Reasoning"
    if "unanswerable" in text or "answer the following question" in text:
        return "Reasoning"
    return "Reasoning"


def count_tokens(question: str, llm_model: str) -> int:
    if llm_model in {"gpt-4-0613", "gpt-4o"}:
        return len(GPT4_ENCODING.encode(question))
    return int(len(question.split()) * 1.25)


def extract_question_features(question: str, llm_model: str) -> dict:
    words = question.split()
    word_count = len(words)
    return {
        "question_length_words": word_count,
        "question_length_chars": len(question),
        "question_complexity_score": round(textstat.flesch_kincaid_grade(question), 2),
        "has_few_shot_examples": bool(FEW_SHOT_PATTERN.search(question)),
        "prompt_contains_system_instructions": bool(SYSTEM_INSTRUCTION_PATTERN.search(question)),
        "question_category": classify_question_category(question),
        "is_ambiguous": word_count < 5,
        "contains_negation": bool(NEGATION_PATTERN.search(question)),
        "context_token_count": count_tokens(question, llm_model),
    }


def enrich_question_features(df: pd.DataFrame) -> pd.DataFrame:
    if "llm_model" not in df.columns:
        raise ValueError("Expected llm_model column for context_token_count")

    existing = [col for col in QUESTION_FEATURE_COLUMNS if col in df.columns]
    base = df.drop(columns=existing).reset_index(drop=True)
    feature_df = base.apply(
        lambda row: extract_question_features(row["question"], row["llm_model"]),
        axis=1,
        result_type="expand",
    )
    return pd.concat([base, feature_df[QUESTION_FEATURE_COLUMNS]], axis=1)


def main():
    input_path = INPUT_PATH if INPUT_PATH.exists() else FALLBACK_INPUT_PATH
    df = pd.read_csv(input_path)
    enriched = enrich_question_features(df)
    enriched.to_csv(OUTPUT_PATH, index=False)
    print(f"Read {input_path}")
    print(f"Saved {len(enriched)} rows and {len(enriched.columns)} columns to {OUTPUT_PATH}")
    print()
    print(enriched[QUESTION_FEATURE_COLUMNS].describe(include="all").to_string())
    print()
    print(enriched["question_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
