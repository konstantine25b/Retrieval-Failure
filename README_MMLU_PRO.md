# MMLU-Pro — Data & Models

Standalone documentation for the MMLU-Pro branch of the Retrieval Failure project: raw data sources, enriched training data, feature selection, and trained classifiers (XGBoost, Random Forest, and MLP).

Paper: [MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark](https://arxiv.org/abs/2406.01574) (NeurIPS 2024)

---

## 1. Source data (what we had)

We did **not** use the question bank alone. Each row is one **model–question outcome**: whether a specific LLM answered a specific MMLU-Pro question correctly.

| Source | Location | Description |
|--------|----------|-------------|
| Questions | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | ~12,032 multiple-choice questions, 10 options (A–J) |
| Model predictions | [TIGER-AI-Lab/MMLU-Pro eval_results](https://github.com/TIGER-AI-Lab/MMLU-Pro/tree/main/eval_results) | Official cached outputs for 47 LLMs (5-shot eval) |

Downloaded locally to `data/mmlu_pro/eval_results/` via:

```bash
python scripts/download_mmlu_pro.py
```

Each JSON file contains per-question fields: `question`, `options`, `answer` (gold), `pred` (model choice), `category`, `question_id`.

---

## 2. Built dataset (`mmlu_pro_full.csv`)

```bash
python scripts/build_mmlu_pro_dataset.py
```

**Output:** `data/mmlu_pro_full.csv`

| | Value |
|--|-------|
| Rows | **563,787** |
| Models | **47** |
| Questions per model | ~12,032 |
| `no_error` (correct) | 280,943 (49.8%) |
| `error` (wrong / no answer) | 282,844 (50.2%) |

### Columns

| Column | Description |
|--------|-------------|
| `question` | Question stem + options A–J (newline-separated) |
| `llm_model` | Model that answered (e.g. `gpt-4o-2024-08-06`, `Llama-2-7b-hf`) |
| `error` | `no_error` if model answer matches gold, else `error` |
| `id` | MMLU-Pro `question_id` |
| `category` | Subject (e.g. `math`, `physics`, `law`) |
| `gold_answer` | Correct option letter (A–J) |
| `model_answer` | Model's chosen letter, or empty if no answer |

Each question appears once per model. The same `id` can appear up to 47 times with different `llm_model` / `error` values.

### Example — correct row

```
question:   What will be the number of lamps, each having 300 lumens...
            A. 6  B. 1  C. 2  ...
llm_model:  DeepSeek-Coder-V2
error:      no_error
category:   engineering
gold_answer: C
model_answer: C
```

### Example — wrong row

```
question:   In force-current analogy, electrical analogous quantity...
            A. inductance.  ...  I. flux.  J. current.
llm_model:  DeepSeek-Coder-V2
error:      error
category:   engineering
gold_answer: I
model_answer: G
```

### All 47 models

`DeepSeek-Coder-V2`, `Llama-2-13b-hf`, `Llama-2-70b-hf`, `Llama-2-7b-hf`, `Meta-Llama-3-70B`, `Meta-Llama-3-70B-Instruct`, `Meta-Llama-3-8B`, `Meta-Llama-3-8B-Instruct`, `Meta-Llama-3_1-70B`, `Meta-Llama-3_1-70B-Instruct`, `Meta-Llama-3_1-8B`, `Meta-Llama-3_1-8B-Instruct`, `Mistral-7B-Instruct-v0.1`, `Mistral-7B-Instruct-v0.2`, `Mistral-7B-v0.1`, `Mistral-7B-v0.2-hf`, `Mixtral-8x7B-Instruct-v0.1`, `Mixtral-8x7B-v0.1`, `Phi-3-mini-4k-instruct`, `Qwen1.5-110B`, `Qwen1.5-14B-Chat`, `Qwen1.5-72B-Chat`, `Qwen1.5-7B-Chat`, `Yi-34B`, `Yi-6B`, `Yi-6b-Chat`, `arx_0314`, `arx_3`, `c4ai-command-r-v01`, `claude-3-5-haiku-20241022`, `claude-3-5-sonnet-20241022`, `claude-3.5-sonnet`, `deepseek`, `deepseek-chat-v2_5`, `flash_0shots_00_35_03`, `gemini-1.5-flash-002`, `gemini-1.5-pro-002`, `gemma-7b`, `gpt-4o-2024-08-06`, `gpt-4o-mini`, `gpt4o(2024-05-13)`, `iask_pro`, `jamba-1.5-large`, `mathstral-7B`, `opus_2shots_00_37_14`, `sonnet-3.5_0shots_09_34_29`, `sonnet_0shots_12_01_18`

---

## 3. Enriched dataset (what we used for training)

```bash
python scripts/enrich_mmlu_pro_full.py
```

**Output:** `data/mmlu_pro_full_enriched.csv` — **563,787 rows × 35 columns**

This is the **only dataset used** to train MMLU-Pro models. It is standalone (not merged with ReaLMistake or Mis-prompt).

### Target & identifiers (3 columns)

| Column | Role |
|--------|------|
| `question` | Input text (not used directly by the model) |
| `llm_model` | Model identifier |
| `error` | Target: `error` vs `no_error` |

### Model-level features (23 columns)

Joined by `llm_model` from `scripts/model_features.py`. Each model has one fixed feature row.

| Column | Examples |
|--------|----------|
| `model_name` | Normalized model name |
| `context_window_tokens` | 4096 (Llama-2-7b) · 131072 (DeepSeek-Coder-V2) · 128000 (GPT-4o) |
| `attention_type` | MHA · GQA · MoE · hybrid_Mamba |
| `tokenizer_type` | sentencepiece_BPE · cl100k_BPE · bytelevel_BPE |
| `positional_encoding_type` | RoPE · learned_absolute |
| `is_open_source`, `multilingual_support` | Boolean flags |
| `temperature`, `top_p`, `top_k`, `repetition_penalty`, … | API / inference hyperparameters |
| `galileo_qa_no_rag`, `galileo_qa_with_rag`, `galileo_longform` | External benchmark proxies |
| `crag_hallucination_rate`, `crag_accuracy` | CRAG benchmark proxies |
| `vocab_size`, `knowledge_cutoff_year`, `max_output_tokens`, … | Architecture metadata |

### Question-level features (9 columns)

Computed per row from `question` text (via `scripts/enrich_question_features.py):

| Column | Method |
|--------|--------|
| `question_length_words` | Word count |
| `question_length_chars` | Character count |
| `question_complexity_score` | Flesch-Kincaid grade (`textstat`) |
| `question_category` | MMLU subject from dataset `category` field |
| `context_token_count` | `tiktoken` for API models, word × 1.25 for open-weight |
| `contains_negation` | Regex flag |
| `has_few_shot_examples` | Regex flag |
| `prompt_contains_system_instructions` | Regex flag |
| `is_ambiguous` | Regex flag |

**Total engineered features for modeling:** 32 (17 numeric + 6 boolean + 5 categorical → WOE)

---

## 4. Modeling overview

All three models solve the same task: **binary classification** — predict whether a given model will get a given question wrong (`error` = 1) vs correct (`no_error` = 0).

They share an identical preprocessing pipeline (stratified split → WOE → IV → RFE → train → evaluate). The only differences are the **RFE base estimator** (matches the final model family) and the **final classifier hyperparameters**.

| | XGBoost | Random Forest | MLP |
|--|---------|---------------|-----|
| Variant | WOE + IV + RFE + complex XGBoost | WOE + IV + RFE + complex Random Forest | WOE + IV + RFE + complex MLP |
| Training script | `scripts/train_xgboost_mmlu_pro_woe_iv_rfe_complex.py` | `scripts/train_random_forest_mmlu_pro_woe_iv_rfe_complex.py` | `scripts/train_mlp_mmlu_pro_woe_iv_rfe_complex.py` |
| Notebook | `notebooks/train_xgboost_mmlu_pro_woe_iv_rfe_complex.ipynb` | `notebooks/train_random_forest_mmlu_pro_woe_iv_rfe_complex.ipynb` | `notebooks/train_mlp_mmlu_pro_woe_iv_rfe_complex.ipynb` |
| Saved model | `models/xgboost_mmlu_pro_woe_iv_rfe_complex.json` | `models/random_forest_mmlu_pro_woe_iv_rfe_complex.joblib` | `models/mlp_mmlu_pro_woe_iv_rfe_complex.joblib` |
| Preprocessing | `models/xgboost_mmlu_pro_woe_iv_rfe_complex_preprocessing.json` | `models/random_forest_mmlu_pro_woe_iv_rfe_complex_preprocessing.json` | `models/mlp_mmlu_pro_woe_iv_rfe_complex_preprocessing.json` |
| Metrics | `models/xgboost_mmlu_pro_woe_iv_rfe_complex_metrics.json` | `models/random_forest_mmlu_pro_woe_iv_rfe_complex_metrics.json` | `models/mlp_mmlu_pro_woe_iv_rfe_complex_metrics.json` |
| Test accuracy | **0.730** | 0.685 | 0.685 |
| Test ROC-AUC | **0.808** | 0.746 | 0.748 |

---

## 5. Shared training pipeline (Steps 1–3)

These steps are identical for all three models.

### Step 1 — Load data & stratified split

| Split | Rows | Share |
|-------|------|-------|
| Train | 394,650 | 70% |
| Validation | 84,568 | 15% |
| Test | 84,569 | 15% |

- **Random state:** 42
- **Stratified** on target (`error` vs `no_error`)
- Class balance is ~50/50 in all splits

---

### Step 2 — WOE encoding (Weight of Evidence)

Applied on **train split only** (maps frozen for val/test):

| Categorical column | WOE output |
|--------------------|------------|
| `model_name` | `model_name_woe` |
| `positional_encoding_type` | `positional_encoding_type_woe` |
| `attention_type` | `attention_type_woe` |
| `tokenizer_type` | `tokenizer_type_woe` |
| `question_category` | `question_category_woe` |

WOE maps encode how much each category shifts the log-odds of error vs no_error. Example: `Llama-2-7b-hf` → WOE +1.50 (higher error rate); `claude-3-5-sonnet-20241022` → WOE −1.25 (lower error rate).

Missing numeric values imputed with **train medians** before IV/RFE/training.

**Feature matrix size:** 32 columns (17 numeric + 6 boolean + 5 WOE).

---

### Step 3 — Information Value (IV) filtering

IV measures univariate predictive power of each feature (higher = more discriminative).

| Setting | Value |
|---------|-------|
| IV bins (numeric) | 5 quantile bins |
| IV threshold (this run) | **0** (notebook); script default is **0.02** |

#### IV scores (computed on train split, sorted descending)

| Feature | IV | Passes @ 0.02 |
|---------|-----|---------------|
| `model_name_woe` | 0.567 | Yes |
| `crag_hallucination_rate` | 0.511 | Yes |
| `crag_accuracy` | 0.497 | Yes |
| `galileo_qa_no_rag` | 0.492 | Yes |
| `galileo_qa_with_rag` | 0.472 | Yes |
| `galileo_longform` | 0.436 | Yes |
| `top_p` | 0.374 | Yes |
| `temperature` | 0.344 | Yes |
| `positional_encoding_type_woe` | 0.335 | Yes |
| `is_open_source` | 0.335 | Yes |
| `tokenizer_type_woe` | 0.328 | Yes |
| `context_window_tokens` | 0.325 | Yes |
| `max_output_tokens` | 0.228 | Yes |
| `max_tokens_requested` | 0.228 | Yes |
| `attention_type_woe` | 0.201 | Yes |
| `question_category_woe` | 0.147 | Yes |
| `multilingual_support` | 0.068 | Yes |
| `vocab_size` | 0.035 | Yes |
| `question_length_words` | 0.020 | **No** |
| `question_complexity_score` | 0.020 | **No** |
| `question_length_chars` | 0.012 | **No** |
| `context_token_count` | 0.009 | **No** |
| `contains_negation` | 0.001 | **No** |
| `repetition_penalty`, `frequency_penalty`, `has_few_shot_examples`, `prompt_contains_system_instructions`, `is_ambiguous`, `top_k`, `knowledge_cutoff_year`, `presence_penalty`, `stop_sequences_count` | 0.000 | **No** |

**Result (this run):** With `IV_THRESHOLD = 0`, all **32 / 32** features passed to RFE.

**If IV threshold = 0.02 (script default):** **17 / 32** features would pass — mostly model/benchmark features plus `question_category_woe`. Raw question length/complexity would be dropped at this stage.

**Interpretation:** On MMLU-Pro, model identity and benchmark proxies dominate univariate signal. Question text metrics have low IV because the same question repeats across 47 models; subject (`question_category_woe`) matters more than length.

---

## 6. XGBoost model

### Step 4 — RFE (XGBoost estimator)

| Setting | Value |
|---------|-------|
| Estimator | XGBoost (100 trees, max_depth=3, lr=0.1) |
| Features to select | **15** |
| Input pool | 32 IV-passing features |

#### RFE ranking

| Rank | Feature | Selected |
|------|---------|----------|
| 1 | `model_name_woe` | Yes |
| 1 | `contains_negation` | Yes |
| 1 | `context_token_count` | Yes |
| 1 | `question_length_chars` | Yes |
| 1 | `question_complexity_score` | Yes |
| 1 | `question_length_words` | Yes |
| 1 | `max_output_tokens` | Yes |
| 1 | `context_window_tokens` | Yes |
| 1 | `tokenizer_type_woe` | Yes |
| 1 | `question_category_woe` | Yes |
| 1 | `temperature` | Yes |
| 1 | `top_p` | Yes |
| 1 | `crag_hallucination_rate` | Yes |
| 1 | `galileo_qa_with_rag` | Yes |
| 1 | `galileo_longform` | Yes |
| 2 | `galileo_qa_no_rag` | No |
| 3 | `crag_accuracy` | No |
| 4 | `positional_encoding_type_woe` | No |
| 5 | `multilingual_support` | No |
| 6 | `is_open_source` | No |
| 7+ | remaining features | No |

**Result:** **15 features** selected — mix of model identity (`model_name_woe`, `tokenizer_type_woe`), architecture (`context_window_tokens`, `max_output_tokens`), hyperparams (`temperature`, `top_p`), benchmark proxies (`crag_hallucination_rate`, `galileo_qa_with_rag`, `galileo_longform`), and question signals (`question_category_woe`, length/complexity/token counts, `contains_negation`).

RFE kept question features that IV alone ranked weakly, because they add incremental value in combination with model features.

---

### Step 5 — Complex XGBoost training

Trained on the 15 RFE-selected features with early stopping on the validation set.

| Hyperparameter | Value |
|----------------|-------|
| `n_estimators` | 1000 |
| `max_depth` | 7 |
| `learning_rate` | 0.02 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 3 |
| `gamma` | 0.1 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 1.0 |
| `scale_pos_weight` | auto (class balance ~1.0) |
| `early_stopping_rounds` | 50 |
| **Best iteration** | **999** (used all 1000 trees) |

---

### Step 6 — XGBoost evaluation results

#### Summary metrics

| Split | Rows | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|------|----------|-----------|--------|-----|---------|
| **Train** | 394,650 | 0.7423 | 0.7398 | 0.7503 | 0.7450 | 0.8229 |
| **Val** | 84,568 | 0.7310 | 0.7286 | 0.7389 | 0.7337 | 0.8095 |
| **Test** | 84,569 | **0.7302** | **0.7283** | **0.7371** | **0.7327** | **0.8076** |

Train–test gap is modest (~1.2 pp accuracy, ~1.5 pp ROC-AUC), indicating reasonable generalization on held-out model–question pairs.

#### Test confusion matrix

```
                 Predicted
                 no_error   error
Actual no_error    30,476   11,666
Actual error       11,153   31,274
```

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| no_error (0) | 0.73 | 0.72 | 0.73 | 42,142 |
| error (1) | 0.73 | 0.74 | 0.73 | 42,427 |

#### Inference on test set

| Metric | Value |
|--------|-------|
| Rows scored | 84,569 |
| Positive predictions (`error`) | 42,940 |
| Mean predicted probability | 0.500 |

The model is well-calibrated at the aggregate level (mean probability ≈ 0.50, matching ~50% base rate).

---

### XGBoost final feature set

These 15 features are in the saved model and preprocessing artifact:

1. `model_name_woe`
2. `contains_negation`
3. `context_token_count`
4. `question_length_chars`
5. `question_complexity_score`
6. `question_length_words`
7. `max_output_tokens`
8. `context_window_tokens`
9. `tokenizer_type_woe`
10. `question_category_woe`
11. `temperature`
12. `top_p`
13. `crag_hallucination_rate`
14. `galileo_qa_with_rag`
15. `galileo_longform`

**Breakdown:** 6 question-derived · 5 model identity/architecture · 2 hyperparams · 2 benchmark proxies (CRAG/Galileo)

#### Top feature importances (XGBoost)

| Feature | Importance |
|---------|------------|
| `model_name_woe` | 0.382 |
| `crag_hallucination_rate` | 0.146 |
| `question_category_woe` | 0.092 |
| `top_p` | 0.091 |
| `galileo_longform` | 0.049 |
| `question_length_words` | 0.037 |
| `question_length_chars` | 0.036 |
| `question_complexity_score` | 0.034 |
| `contains_negation` | 0.032 |
| `context_token_count` | 0.029 |

---

## 7. Random Forest model

The Random Forest experiment mirrors the XGBoost pipeline exactly through Steps 1–3, then uses a **Random Forest estimator for RFE** and a **complex Random Forest classifier** for the final model. This is the natural parallel setup: RFE ranks features by what the RF family finds useful, not what XGBoost finds useful.

### Step 4 — RFE (Random Forest estimator)

| Setting | Value |
|---------|-------|
| Estimator | Random Forest (100 trees, max_depth=3, class_weight=balanced) |
| Features to select | **15** |
| Input pool | 32 IV-passing features |

#### RFE ranking

| Rank | Feature | Selected |
|------|---------|----------|
| 1 | `model_name_woe` | Yes |
| 1 | `context_token_count` | Yes |
| 1 | `question_complexity_score` | Yes |
| 1 | `question_length_words` | Yes |
| 1 | `vocab_size` | Yes |
| 1 | `is_open_source` | Yes |
| 1 | `positional_encoding_type_woe` | Yes |
| 1 | `question_category_woe` | Yes |
| 1 | `top_p` | Yes |
| 1 | `crag_hallucination_rate` | Yes |
| 1 | `temperature` | Yes |
| 1 | `galileo_qa_no_rag` | Yes |
| 1 | `galileo_qa_with_rag` | Yes |
| 1 | `crag_accuracy` | Yes |
| 1 | `galileo_longform` | Yes |
| 2 | `context_window_tokens` | No |
| 3 | `question_length_chars` | No |
| 4 | `attention_type_woe` | No |
| 5 | `tokenizer_type_woe` | No |
| 6 | `max_tokens_requested` | No |
| 7 | `multilingual_support` | No |
| 8 | `max_output_tokens` | No |
| 9 | `contains_negation` | No |
| 10+ | remaining features | No |

**Result:** **15 features** selected — heavily weighted toward **model identity and benchmark proxies**. Compared to XGBoost RFE, Random Forest RFE:

- **Kept all four Galileo/CRAG benchmark features** (`galileo_qa_no_rag`, `galileo_qa_with_rag`, `crag_accuracy`, `galileo_longform`) plus `crag_hallucination_rate`
- **Dropped** `contains_negation`, `question_length_chars`, `context_window_tokens`, `max_output_tokens`, and `tokenizer_type_woe`
- **Added** `vocab_size`, `is_open_source`, and `positional_encoding_type_woe`

This reflects a different inductive bias: RF RFE favors stable model-level signals (benchmark scores, architecture metadata) over per-question text micro-features that XGBoost exploited more effectively.

---

### Step 5 — Complex Random Forest training

Trained on the 15 RFE-selected features. No early stopping (standard for bagging ensembles).

| Hyperparameter | Value |
|----------------|-------|
| `n_estimators` | 300 |
| `max_depth` | 7 |
| `min_samples_split` | 5 |
| `min_samples_leaf` | 2 |
| `max_features` | 0.8 |
| `class_weight` | balanced |
| `random_state` | 42 |
| `n_jobs` | -1 (all cores) |

**Design rationale:** `max_depth=7` matches the XGBoost tree depth for comparability. `max_features=0.8` parallels XGBoost's `colsample_bytree=0.8`. `class_weight=balanced` handles the ~50/50 label split similarly to XGBoost's `scale_pos_weight`. Fewer trees (300 vs 1000) because each RF tree is trained on bootstrap samples in parallel rather than sequentially boosted residuals.

**Saved format:** `joblib` (`.joblib`), not JSON — scikit-learn's native serialization for sklearn estimators.

---

### Step 6 — Random Forest evaluation results

#### Summary metrics

| Split | Rows | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|------|----------|-----------|--------|-----|---------|
| **Train** | 394,650 | 0.6890 | 0.6906 | 0.6884 | 0.6895 | 0.7514 |
| **Val** | 84,568 | 0.6872 | 0.6888 | 0.6868 | 0.6878 | 0.7498 |
| **Test** | 84,569 | **0.6848** | **0.6864** | **0.6843** | **0.6853** | **0.7463** |

Train–test gap is very small (~0.4 pp accuracy), suggesting the RF ensemble is not overfitting heavily — it simply has lower capacity to capture the signal in this dataset compared to gradient boosting.

#### Test confusion matrix

```
                 Predicted
                 no_error   error
Actual no_error    28,878   13,264
Actual error       13,396   29,031
```

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| no_error (0) | 0.68 | 0.69 | 0.68 | 42,142 |
| error (1) | 0.69 | 0.68 | 0.69 | 42,427 |

#### Inference on test set

| Metric | Value |
|--------|-------|
| Rows scored | 84,569 |
| Positive predictions (`error`) | 42,295 |
| Mean predicted probability | 0.500 |

Aggregate calibration is also good (mean probability ≈ 0.50).

---

### Random Forest final feature set

These 15 features are in the saved model and preprocessing artifact:

1. `model_name_woe`
2. `context_token_count`
3. `question_complexity_score`
4. `question_length_words`
5. `vocab_size`
6. `is_open_source`
7. `positional_encoding_type_woe`
8. `question_category_woe`
9. `top_p`
10. `crag_hallucination_rate`
11. `temperature`
12. `galileo_qa_no_rag`
13. `galileo_qa_with_rag`
14. `crag_accuracy`
15. `galileo_longform`

**Breakdown:** 3 question-derived · 4 model identity/architecture · 2 hyperparams · 6 benchmark proxies (Galileo + CRAG)

Only **3 features overlap** in rank-1 selection between the two tree models: `model_name_woe`, `question_category_woe`, and `crag_hallucination_rate` (plus shared hyperparams `temperature` and `top_p`).

---

## 8. MLP (Neural Network) model

The MLP experiment mirrors the same pipeline through Steps 1–3, then uses a **small MLP for RFE** and a **StandardScaler + MLPClassifier pipeline** for the final model. Features are scaled before training because neural nets are sensitive to input magnitude; tree models do not need this step.

### Step 4 — RFE (MLP estimator)

| Setting | Value |
|---------|-------|
| Estimator | MLP (1 hidden layer × 32 units, max_iter=40, Adam) |
| Importance | Mean absolute first-layer weights per input feature |
| Features to select | **15** |
| Input pool | 32 IV-passing features |

#### RFE ranking

| Rank | Feature | Selected |
|------|---------|----------|
| 1 | `model_name_woe` | Yes |
| 1 | `context_token_count` | Yes |
| 1 | `question_length_chars` | Yes |
| 1 | `question_complexity_score` | Yes |
| 1 | `question_length_words` | Yes |
| 1 | `vocab_size` | Yes |
| 1 | `attention_type_woe` | Yes |
| 1 | `is_open_source` | Yes |
| 1 | `question_category_woe` | Yes |
| 1 | `crag_accuracy` | Yes |
| 1 | `crag_hallucination_rate` | Yes |
| 1 | `positional_encoding_type_woe` | Yes |
| 1 | `galileo_qa_no_rag` | Yes |
| 1 | `galileo_longform` | Yes |
| 1 | `top_p` | Yes |
| 2 | `multilingual_support` | No |
| 3 | `temperature` | No |
| 4 | `galileo_qa_with_rag` | No |
| 5 | `repetition_penalty` | No |
| 6 | `max_tokens_requested` | No |
| 7 | `max_output_tokens` | No |
| 8 | `knowledge_cutoff_year` | No |
| 9 | `top_k` | No |
| 10 | `stop_sequences_count` | No |
| 11 | `context_window_tokens` | No |
| 12 | `contains_negation` | No |
| 13 | `tokenizer_type_woe` | No |
| 14+ | remaining features | No |

**Result:** **15 features** selected — very close to Random Forest RFE, with two swaps:

- **Kept** `question_length_chars` and `attention_type_woe` (question/architecture signals RF RFE dropped)
- **Dropped** `temperature` and `galileo_qa_with_rag` (RF RFE kept both)

Overall the MLP RFE set still leans on model identity and benchmark proxies, but retains slightly more question-level and architecture detail than RF RFE.

---

### Step 5 — Complex MLP training

Trained on the 15 RFE-selected features inside a `StandardScaler → MLPClassifier` pipeline. Early stopping uses 10% of the training data as an internal validation holdout.

| Hyperparameter | Value |
|----------------|-------|
| Hidden layers | (64, 32) |
| Activation | ReLU |
| Solver | Adam |
| `alpha` (L2) | 0.0001 |
| `batch_size` | 4096 |
| `learning_rate_init` | 0.001 |
| `learning_rate` | adaptive |
| `max_iter` | 200 |
| `early_stopping` | true |
| `validation_fraction` | 0.1 |
| `n_iter_no_change` | 15 |
| **Training iterations** | **50** (early stopped) |
| **Final loss** | **0.589** |

**Design rationale:** A shallow 2-layer MLP (64 → 32 → 1) is enough capacity for 15 tabular features without overfitting. `StandardScaler` is required because WOE and raw numeric features live on different scales. RFE uses a smaller single-hidden-layer MLP (32 units, 40 iterations) as a lightweight proxy for the final network family.

**Saved format:** `joblib` (`.joblib`) — scikit-learn pipeline with scaler + MLP.

---

### Step 6 — MLP evaluation results

#### Summary metrics

| Split | Rows | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|------|----------|-----------|--------|-----|---------|
| **Train** | 394,650 | 0.6890 | 0.6883 | 0.6946 | 0.6914 | 0.7526 |
| **Val** | 84,568 | 0.6875 | 0.6869 | 0.6931 | 0.6900 | 0.7518 |
| **Test** | 84,569 | **0.6850** | **0.6846** | **0.6899** | **0.6872** | **0.7479** |

Train–test gap is ~0.4 pp accuracy, similar to Random Forest — the network generalizes without heavy overfitting but does not reach XGBoost performance.

#### Test confusion matrix

```
                 Predicted
                 no_error   error
Actual no_error    28,657   13,485
Actual error       13,157   29,270
```

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| no_error (0) | 0.69 | 0.68 | 0.68 | 42,142 |
| error (1) | 0.68 | 0.69 | 0.69 | 42,427 |

#### Inference on test set

| Metric | Value |
|--------|-------|
| Rows scored | 84,569 |
| Positive predictions (`error`) | 42,755 |
| Mean predicted probability | 0.501 |

Aggregate calibration is good (mean probability ≈ 0.50).

---

### MLP final feature set

These 15 features are in the saved model and preprocessing artifact:

1. `model_name_woe`
2. `context_token_count`
3. `question_length_chars`
4. `question_complexity_score`
5. `question_length_words`
6. `vocab_size`
7. `attention_type_woe`
8. `is_open_source`
9. `question_category_woe`
10. `crag_accuracy`
11. `crag_hallucination_rate`
12. `positional_encoding_type_woe`
13. `galileo_qa_no_rag`
14. `galileo_longform`
15. `top_p`

**Breakdown:** 4 question-derived · 5 model identity/architecture · 1 hyperparam · 5 benchmark proxies (Galileo + CRAG)

**13 / 15 features overlap** with Random Forest RFE; MLP swaps in `question_length_chars` and `attention_type_woe` for RF's `temperature` and `galileo_qa_with_rag`.

#### Top input importances (MLP first-layer weights)

| Feature | Importance |
|---------|------------|
| `question_category_woe` | 0.230 |
| `context_token_count` | 0.189 |
| `question_length_chars` | 0.175 |
| `question_complexity_score` | 0.169 |
| `question_length_words` | 0.163 |
| `model_name_woe` | 0.162 |
| `is_open_source` | 0.145 |
| `galileo_longform` | 0.137 |
| `positional_encoding_type_woe` | 0.136 |
| `crag_accuracy` | 0.134 |

Unlike XGBoost (where `model_name_woe` dominates at 0.38), the MLP spreads weight more evenly — `question_category_woe` is the strongest input, reflecting the network's reliance on subject difficulty alongside model metadata.

---

## 9. XGBoost vs Random Forest vs MLP — head-to-head

All three models trained on the same 563,787-row dataset, same 70/15/15 stratified split (random_state=42), same WOE maps, same IV threshold (0), and same RFE target count (15). Differences are the RFE estimator family and final classifier.

| Metric | XGBoost (test) | Random Forest (test) | MLP (test) |
|--------|----------------|----------------------|------------|
| Accuracy | **0.730** | 0.685 | 0.685 |
| Precision | **0.728** | 0.686 | 0.685 |
| Recall | **0.737** | 0.684 | 0.690 |
| F1 | **0.733** | 0.685 | 0.687 |
| ROC-AUC | **0.808** | 0.746 | 0.748 |
| False positives (test) | 11,666 | 13,264 | 13,485 |
| False negatives (test) | 11,153 | 13,396 | 13,157 |

**Why XGBoost wins on MMLU-Pro:**

1. **Feature interactions:** Gradient boosting sequentially corrects residuals and captures non-linear interactions between model identity and question-level features (e.g. `contains_negation` × `model_name_woe`). RF and MLP miss some of these joint effects.
2. **RFE feature set:** XGBoost RFE retained more question-level signal (`contains_negation`, length chars, token counts) that helps distinguish hard questions within the same model tier. RF and MLP RFE converged on benchmark proxies.
3. **Scale:** 1000 boosted trees with early stopping vs 300 bagged trees or a 64→32 MLP — boosting had more capacity to fit the ~395k training rows without severe overfitting (train-test gap ~1.2 pp for XGBoost vs ~0.4 pp for RF/MLP, but the tree/neural baselines hit a lower ceiling).

**Random Forest vs MLP:**

- Test accuracy is effectively tied (~0.685); MLP has marginally higher ROC-AUC (+0.2 pp) and recall (+0.6 pp).
- MLP RFE keeps `question_length_chars` and `attention_type_woe`; RF RFE keeps `temperature` and `galileo_qa_with_rag`.
- MLP requires feature scaling and longer per-epoch training; RF trains faster on CPU with `n_jobs=-1`.

**When RF or MLP might still be useful:**

- Faster or simpler baselines (RF: parallel trees; MLP: pure scikit-learn, no XGBoost)
- RF: per-tree split interpretability for stakeholder review
- MLP: tests whether a neural net on tabular WOE features adds value over bagging — on this task it does not beat RF meaningfully and both trail XGBoost by ~4.5 pp accuracy

---

## 10. Comparison with other project models

| Model | Dataset | Test rows | Test accuracy | Test ROC-AUC |
|-------|---------|-----------|---------------|--------------|
| **MMLU-Pro complex XGBoost** | `mmlu_pro_full_enriched.csv` | 84,569 | **0.730** | **0.808** |
| MMLU-Pro complex MLP | `mmlu_pro_full_enriched.csv` | 84,569 | 0.685 | 0.748 |
| MMLU-Pro complex Random Forest | `mmlu_pro_full_enriched.csv` | 84,569 | 0.685 | 0.746 |
| Combined complex XGBoost | `combined_full_enriched.csv` | 4,544 | 0.755 | 0.832 |

MMLU-Pro test set is ~19× larger. Accuracy is slightly lower because MMLU error/no_error is a near-random 50/50 split per model capability tier, while ReaLMistake/Mis-prompt in the combined set have different label dynamics.

---

## 11. Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/view_mmlu_pro_enriched.ipynb` | Explore enriched data: null counts, per-model samples, feature distributions |
| `notebooks/train_xgboost_mmlu_pro_woe_iv_rfe_complex.ipynb` | Full XGBoost pipeline: split → WOE → IV → RFE → XGBoost → metrics → save artifacts |
| `notebooks/train_random_forest_mmlu_pro_woe_iv_rfe_complex.ipynb` | Full Random Forest pipeline: split → WOE → IV → RFE → Random Forest → metrics → save artifacts |
| `notebooks/train_mlp_mmlu_pro_woe_iv_rfe_complex.ipynb` | Full MLP pipeline: split → WOE → IV → RFE → StandardScaler → MLP → metrics → save artifacts |

```bash
jupyter notebook notebooks/
```

---

## 12. End-to-end reproduction

### Data pipeline (shared)

```bash
source venv/bin/activate

python scripts/download_mmlu_pro.py
python scripts/build_mmlu_pro_dataset.py
python scripts/enrich_mmlu_pro_full.py
```

### Train XGBoost

```bash
python scripts/train_xgboost_mmlu_pro_woe_iv_rfe_complex.py
```

Or interactively: `notebooks/train_xgboost_mmlu_pro_woe_iv_rfe_complex.ipynb`

### Train Random Forest

```bash
python scripts/train_random_forest_mmlu_pro_woe_iv_rfe_complex.py
```

Or interactively: `notebooks/train_random_forest_mmlu_pro_woe_iv_rfe_complex.ipynb`

### Train MLP

```bash
python scripts/train_mlp_mmlu_pro_woe_iv_rfe_complex.py
```

Or interactively: `notebooks/train_mlp_mmlu_pro_woe_iv_rfe_complex.ipynb`

**Expected runtime:** Random Forest ~3 minutes on 394k training rows with `n_jobs=-1` (M-series Mac / multi-core CPU). MLP ~5–10 minutes (50 early-stopped epochs, batch_size=4096). XGBoost takes longer due to 1000 sequential boosting rounds.

---

## 13. Key takeaways

1. **563,787 model–question rows** from 47 LLMs on MMLU-Pro, labeled error vs no_error.
2. **32 engineered features** (model specs + question metrics) → WOE on 5 categoricals.
3. **IV + RFE** reduced to **15 features** per model; model identity and benchmark proxies are strongest univariate signals.
4. **XGBoost (best):** 73.0% test accuracy, 0.808 ROC-AUC — captures question-level interactions that RF and MLP miss.
5. **Random Forest (baseline):** 68.5% test accuracy, 0.746 ROC-AUC — solid bagging baseline, ~4.5 pp behind boosting.
6. **MLP (neural baseline):** 68.5% test accuracy, 0.748 ROC-AUC — matches RF; spreads importance across question features rather than concentrating on `model_name_woe`.
7. All models predict **whether a specific LLM will fail on a specific question** using model metadata and lightweight question features — not the question text itself.
8. RFE feature sets **differ by estimator family**; for strict apples-to-apples model comparison, fix the same 15 features and retrain all classifiers.

---

## References

- Wang et al., [MMLU-Pro](https://arxiv.org/abs/2406.01574) (NeurIPS 2024)
- Dataset: [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
- Eval results: [TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro)
