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

GPT4_ENCODING = None


def _get_gpt4_encoding():
    global GPT4_ENCODING
    if GPT4_ENCODING is None:
        GPT4_ENCODING = tiktoken.get_encoding("cl100k_base")
    return GPT4_ENCODING


CL100K_MODELS = {
    "gpt-4-0613",
    "gpt-4o",
    "gpt-4o-2024-08-06",
    "gpt4o(2024-05-13)",
    "gpt-4o-mini",
    "Phi-3-mini-4k-instruct",
    "arx_0314",
    "arx_3",
    "iask_pro",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-3.5-sonnet",
    "opus_2shots_00_37_14",
    "sonnet-3.5_0shots_09_34_29",
    "sonnet_0shots_12_01_18",
    "flash_0shots_00_35_03",
    "gemini-1.5-flash-002",
    "gemini-1.5-pro-002",
}

MMLU_CATEGORY_LABELS = {
    "math": "Math",
    "physics": "Physics",
    "chemistry": "Chemistry",
    "biology": "Biology",
    "computer science": "Computer Science",
    "engineering": "Engineering",
    "economics": "Economics",
    "business": "Business",
    "health": "Health",
    "psychology": "Psychology",
    "law": "Law",
    "philosophy": "Philosophy",
    "history": "History",
    "other": "Other",
}


def classify_question_category(question: str, category: str | None = None) -> str:
    if category is not None and pd.notna(category):
        normalized = str(category).strip().lower()
        return MMLU_CATEGORY_LABELS.get(normalized, str(category).strip().title())
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
    if llm_model in CL100K_MODELS:
        return len(_get_gpt4_encoding().encode(question))
    return int(len(question.split()) * 1.25)


def extract_question_features(
    question: str,
    llm_model: str,
    category: str | None = None,
) -> dict:
    words = question.split()
    word_count = len(words)
    return {
        "question_length_words": word_count,
        "question_length_chars": len(question),
        "question_complexity_score": round(textstat.flesch_kincaid_grade(question), 2),
        "has_few_shot_examples": bool(FEW_SHOT_PATTERN.search(question)),
        "prompt_contains_system_instructions": bool(SYSTEM_INSTRUCTION_PATTERN.search(question)),
        "question_category": classify_question_category(question, category),
        "is_ambiguous": word_count < 5,
        "contains_negation": bool(NEGATION_PATTERN.search(question)),
        "context_token_count": count_tokens(question, llm_model),
    }


def enrich_question_features(df: pd.DataFrame) -> pd.DataFrame:
    if "llm_model" not in df.columns:
        raise ValueError("Expected llm_model column for context_token_count")

    existing = [col for col in QUESTION_FEATURE_COLUMNS if col in df.columns]
    base = df.drop(columns=existing).reset_index(drop=True)
    category_col = "category" if "category" in base.columns else None
    questions = base["question"].astype(str)

    feature_df = pd.DataFrame(index=base.index)
    feature_df["question_length_words"] = questions.str.split().str.len()
    feature_df["question_length_chars"] = questions.str.len()
    feature_df["question_complexity_score"] = questions.map(
        lambda q: round(textstat.flesch_kincaid_grade(q), 2)
    )
    feature_df["has_few_shot_examples"] = questions.str.contains(FEW_SHOT_PATTERN, regex=True)
    feature_df["prompt_contains_system_instructions"] = questions.str.contains(
        SYSTEM_INSTRUCTION_PATTERN, regex=True
    )
    if category_col:
        feature_df["question_category"] = base[category_col].map(
            lambda c: classify_question_category("", c)
        )
    else:
        feature_df["question_category"] = questions.map(classify_question_category)
    feature_df["is_ambiguous"] = feature_df["question_length_words"] < 5
    feature_df["contains_negation"] = questions.str.contains(NEGATION_PATTERN, regex=True)
    feature_df["context_token_count"] = base.apply(
        lambda row: count_tokens(row["question"], row["llm_model"]),
        axis=1,
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
