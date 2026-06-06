from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "realmistake_full.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "realmistake_full_enriched.csv"

MODEL_FEATURES = {
    "gpt-4-0613": {
        "model_name": "gpt-4-0613",
        "context_window_tokens": 8192,
        "max_output_tokens": 4096,
        "vocab_size": 100277,
        "positional_encoding_type": "learned_absolute",
        "attention_type": "MHA",
        "tokenizer_type": "cl100k_BPE",
        "is_open_source": False,
        "knowledge_cutoff_year": 2021,
        "multilingual_support": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": pd.NA,
        "repetition_penalty": pd.NA,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens_requested": 4096,
        "stop_sequences_count": 0,
        "galileo_qa_no_rag": 0.77,
        "galileo_qa_with_rag": 0.76,
        "galileo_longform": 0.83,
        "crag_hallucination_rate": 0.135,
        "crag_accuracy": 0.335,
    },
    "meta-llama/Llama-2-70b-chat-hf": {
        "model_name": "llama-2-70b-chat-hf",
        "context_window_tokens": 4096,
        "max_output_tokens": 2048,
        "vocab_size": 32000,
        "positional_encoding_type": "RoPE",
        "attention_type": "GQA",
        "tokenizer_type": "sentencepiece_BPE",
        "is_open_source": True,
        "knowledge_cutoff_year": 2022,
        "multilingual_support": False,
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "frequency_penalty": pd.NA,
        "presence_penalty": pd.NA,
        "max_tokens_requested": 1024,
        "stop_sequences_count": 1,
        "galileo_qa_no_rag": 0.65,
        "galileo_qa_with_rag": 0.68,
        "galileo_longform": 0.82,
        "crag_hallucination_rate": 0.287,
        "crag_accuracy": 0.223,
    },
}

FEATURE_COLUMNS = [
    "model_name",
    "context_window_tokens",
    "max_output_tokens",
    "vocab_size",
    "positional_encoding_type",
    "attention_type",
    "tokenizer_type",
    "is_open_source",
    "knowledge_cutoff_year",
    "multilingual_support",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "max_tokens_requested",
    "stop_sequences_count",
    "galileo_qa_no_rag",
    "galileo_qa_with_rag",
    "galileo_longform",
    "crag_hallucination_rate",
    "crag_accuracy",
]


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    unknown = sorted(set(df["llm_model"]) - set(MODEL_FEATURES))
    if unknown:
        raise ValueError(f"Unknown llm_model values: {unknown}")

    feature_df = df["llm_model"].map(MODEL_FEATURES).apply(pd.Series)
    enriched = pd.concat([df.reset_index(drop=True), feature_df[FEATURE_COLUMNS]], axis=1)
    return enriched


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
