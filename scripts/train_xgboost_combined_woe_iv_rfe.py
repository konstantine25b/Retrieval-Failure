import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFE
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
DATA_PATH = PROJECT_ROOT / "data" / "combined_full_enriched.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_combined_woe_iv_rfe.json"
METRICS_PATH = PROJECT_ROOT / "models" / "xgboost_combined_woe_iv_rfe_metrics.json"
PREPROCESSING_PATH = PROJECT_ROOT / "models" / "xgboost_combined_woe_iv_rfe_preprocessing.json"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
IV_THRESHOLD = 0.02
RFE_N_FEATURES = 15
NUMERIC_IV_BINS = 5

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


def compute_iv_grouped(frame: pd.DataFrame, group_col: str, target: str, observed: bool = False) -> float:
    events = frame[target].sum()
    non_events = len(frame) - events
    if events == 0 or non_events == 0:
        return 0.0
    grouped = frame.groupby(group_col, dropna=False, observed=observed)[target].agg(["sum", "count"])
    iv = 0.0
    for _, row in grouped.iterrows():
        bad_dist = row["sum"] / events
        good_dist = (row["count"] - row["sum"]) / non_events
        if bad_dist <= 0 or good_dist <= 0:
            continue
        woe = np.log(bad_dist / good_dist)
        iv += (bad_dist - good_dist) * woe
    return float(iv)


def compute_feature_iv(
    train_df: pd.DataFrame,
    feature_name: str,
    target: str,
    medians: pd.Series,
) -> float:
    if feature_name.endswith("_woe"):
        raw_col = feature_name.removesuffix("_woe")
        return compute_iv_grouped(train_df, raw_col, target)
    if feature_name in BOOLEAN_COLUMNS:
        return compute_iv_grouped(train_df, feature_name, target)
    filled = train_df[feature_name].fillna(medians.get(feature_name, train_df[feature_name].median()))
    try:
        binned = pd.qcut(filled, q=NUMERIC_IV_BINS, duplicates="drop")
    except ValueError:
        return 0.0
    temp = pd.DataFrame({"bin": binned, target: train_df[target]})
    return compute_iv_grouped(temp, "bin", target, observed=True)


def select_iv_features(
    train_df: pd.DataFrame,
    feature_names: list[str],
    target: str,
    medians: pd.Series,
    threshold: float,
) -> tuple[list[str], dict[str, float]]:
    iv_scores = {
        name: compute_feature_iv(train_df, name, target, medians)
        for name in feature_names
    }
    selected = [name for name, score in iv_scores.items() if score >= threshold]
    if not selected:
        selected = [max(iv_scores, key=iv_scores.get)]
    return selected, iv_scores


def select_rfe_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    iv_features: list[str],
    n_features: int,
    scale_pos_weight: float,
) -> tuple[list[str], dict[str, int]]:
    n_select = min(n_features, len(iv_features))
    estimator = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    selector = RFE(estimator=estimator, n_features_to_select=n_select, step=1)
    selector.fit(x_train[iv_features], y_train)
    ranking = {feat: int(rank) for feat, rank in zip(iv_features, selector.ranking_)}
    selected = [feat for feat in iv_features if ranking[feat] == 1]
    return selected, ranking


def prepare_data():
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

    return (
        train_df,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        feature_names,
        woe_maps,
        medians,
    )


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
    (
        train_df,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        feature_names,
        woe_maps,
        medians,
    ) = prepare_data()

    iv_features, iv_scores = select_iv_features(
        train_df,
        feature_names,
        "target",
        medians,
        IV_THRESHOLD,
    )

    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    rfe_features, rfe_ranking = select_rfe_features(
        x_train,
        y_train,
        iv_features,
        RFE_N_FEATURES,
        scale_pos_weight,
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )

    x_train_sel = x_train[rfe_features]
    x_val_sel = x_val[rfe_features]
    x_test_sel = x_test[rfe_features]

    model.fit(
        x_train_sel,
        y_train,
        eval_set=[(x_val_sel, y_val)],
        verbose=False,
    )

    train_metrics = evaluate(model, x_train_sel, y_train, "train")
    val_metrics = evaluate(model, x_val_sel, y_val, "val")
    test_metrics = evaluate(model, x_test_sel, y_test, "test")

    inference_proba = model.predict_proba(x_test_sel)[:, 1]
    inference_pred = (inference_proba >= 0.5).astype(int)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)

    artifact = {
        "feature_names": rfe_features,
        "iv_threshold": IV_THRESHOLD,
        "iv_scores": iv_scores,
        "iv_selected_features": iv_features,
        "rfe_ranking": rfe_ranking,
        "woe_maps": {k: {str(key): val for key, val in v.items()} for k, v in woe_maps.items()},
        "numeric_medians": medians.to_dict(),
    }
    with PREPROCESSING_PATH.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "combined_full_enriched.csv",
                "feature_selection": {
                    "total_features": len(feature_names),
                    "iv_selected": len(iv_features),
                    "rfe_selected": len(rfe_features),
                    "selected_features": rfe_features,
                },
                "train": {k: v for k, v in train_metrics.items() if k != "classification_report"},
                "val": {k: v for k, v in val_metrics.items() if k != "classification_report"},
                "test": {k: v for k, v in test_metrics.items() if k != "classification_report"},
                "inference_sample": {
                    "n_rows": int(len(x_test)),
                    "positive_predictions": int(inference_pred.sum()),
                    "mean_predicted_probability": float(inference_proba.mean()),
                },
            },
            f,
            indent=2,
        )

    print(f"Dataset: {DATA_PATH.name}")
    print(f"Train rows: {len(x_train)} | Val rows: {len(x_val)} | Test rows: {len(x_test)}")
    print(f"Features: {len(feature_names)} -> IV: {len(iv_features)} -> RFE: {len(rfe_features)}")
    print(f"RFE selected: {rfe_features}")
    print()
    for split, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
        print(
            f"{split.upper():5}  accuracy={metrics['accuracy']:.4f}  precision={metrics['precision']:.4f}  "
            f"recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}  roc_auc={metrics['roc_auc']:.4f}"
        )
        print(f"       confusion_matrix={metrics['confusion_matrix']}")
    print()
    print("TEST classification report:")
    print(test_metrics["classification_report"])
    print()
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
