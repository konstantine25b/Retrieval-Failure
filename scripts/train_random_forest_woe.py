import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "realmistake_full_enriched.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_woe.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "random_forest_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

CATEGORICAL_COLUMNS = [
    "model_name",
    "positional_encoding_type",
    "attention_type",
    "tokenizer_type",
    "question_category",
]

BOOLEAN_COLUMNS = [
    "is_open_source",
    "multilingual_support",
    "has_few_shot_examples",
    "prompt_contains_system_instructions",
    "is_ambiguous",
    "contains_negation",
]

NUMERIC_COLUMNS = [
    "context_window_tokens",
    "max_output_tokens",
    "vocab_size",
    "knowledge_cutoff_year",
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
    "question_length_words",
    "question_length_chars",
    "question_complexity_score",
    "context_token_count",
]


def compute_woe_maps(frame: pd.DataFrame, columns: list[str], target: str) -> dict[str, dict]:
    maps = {}
    total_events = frame[target].sum()
    total_non_events = len(frame) - total_events
    for column in columns:
        grouped = frame.groupby(column, dropna=False)[target].agg(["sum", "count"])
        grouped["non_events"] = grouped["count"] - grouped["sum"]
        woe_map = {}
        for value, row in grouped.iterrows():
            event_rate = (row["sum"] + 0.5) / (total_events + 1.0)
            non_event_rate = (row["non_events"] + 0.5) / (total_non_events + 1.0)
            woe_map[value] = float(np.log(event_rate / non_event_rate))
        maps[column] = woe_map
    return maps


def apply_woe(frame: pd.DataFrame, columns: list[str], maps: dict[str, dict]) -> pd.DataFrame:
    transformed = frame.copy()
    for column in columns:
        default = 0.0
        transformed[f"{column}_woe"] = transformed[column].map(maps[column]).fillna(default)
    return transformed


def build_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    boolean = frame[BOOLEAN_COLUMNS].astype(int)
    woe_cols = [f"{col}_woe" for col in CATEGORICAL_COLUMNS]
    return pd.concat([numeric, boolean, frame[woe_cols]], axis=1)


def prepare_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    df["target"] = (df["error"] == "error").astype(int)

    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=RANDOM_STATE,
        stratify=train_val_df["target"],
    )

    woe_maps = compute_woe_maps(train_df, CATEGORICAL_COLUMNS, "target")
    train_df = apply_woe(train_df, CATEGORICAL_COLUMNS, woe_maps)
    val_df = apply_woe(val_df, CATEGORICAL_COLUMNS, woe_maps)
    test_df = apply_woe(test_df, CATEGORICAL_COLUMNS, woe_maps)

    feature_names = NUMERIC_COLUMNS + BOOLEAN_COLUMNS + [f"{col}_woe" for col in CATEGORICAL_COLUMNS]

    x_train = build_feature_matrix(train_df)
    x_val = build_feature_matrix(val_df)
    x_test = build_feature_matrix(test_df)

    medians = x_train.median(numeric_only=True)
    x_train = x_train.fillna(medians)
    x_val = x_val.fillna(medians)
    x_test = x_test.fillna(medians)

    y_train = train_df["target"]
    y_val = val_df["target"]
    y_test = test_df["target"]

    return x_train, y_train, x_val, y_val, x_test, y_test, feature_names, woe_maps, medians


def evaluate(model, x, y, split_name: str) -> dict:
    proba = model.predict_proba(x)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "split": split_name,
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }
    metrics["confusion_matrix"] = confusion_matrix(y, preds).tolist()
    metrics["classification_report"] = classification_report(y, preds, zero_division=0)
    return metrics


def main():
    x_train, y_train, x_val, y_val, x_test, y_test, feature_names, woe_maps, medians = prepare_data()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    train_metrics = evaluate(model, x_train, y_train, "train")
    val_metrics = evaluate(model, x_val, y_val, "val")
    test_metrics = evaluate(model, x_test, y_test, "test")

    inference_proba = model.predict_proba(x_test)[:, 1]
    inference_pred = (inference_proba >= 0.5).astype(int)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    artifact = {
        "feature_names": feature_names,
        "woe_maps": {k: {str(key): val for key, val in v.items()} for k, v in woe_maps.items()},
        "numeric_medians": medians.to_dict(),
    }
    with (MODEL_PATH.parent / "random_forest_preprocessing.json").open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    results = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "inference_sample": {
            "n_rows": int(len(x_test)),
            "positive_predictions": int(inference_pred.sum()),
            "mean_predicted_probability": float(inference_proba.mean()),
        },
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train": {k: v for k, v in train_metrics.items() if k != "classification_report"},
                "val": {k: v for k, v in val_metrics.items() if k != "classification_report"},
                "test": {k: v for k, v in test_metrics.items() if k != "classification_report"},
                "inference_sample": results["inference_sample"],
            },
            f,
            indent=2,
        )

    print(f"Train rows: {len(x_train)} | Val rows: {len(x_val)} | Test rows: {len(x_test)}")
    print(f"Features: {len(feature_names)}")
    print()
    for split in ("train", "val", "test"):
        m = results[split]
        print(
            f"{split.upper():5}  accuracy={m['accuracy']:.4f}  precision={m['precision']:.4f}  "
            f"recall={m['recall']:.4f}  f1={m['f1']:.4f}  roc_auc={m['roc_auc']:.4f}"
        )
        print(f"       confusion_matrix={m['confusion_matrix']}")
    print()
    print("TEST classification report:")
    print(test_metrics["classification_report"])
    print()
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
