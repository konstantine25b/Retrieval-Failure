# MMLU-Pro — Data & XGBoost Model

Standalone documentation for the MMLU-Pro branch of the Retrieval Failure project: raw data sources, enriched training data, feature selection, and the trained XGBoost classifier.

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

This is the **only dataset used** to train the MMLU-Pro XGBoost model. It is standalone (not merged with ReaLMistake or Mis-prompt).

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

Computed per row from `question` text (via `scripts/enrich_question_features.py`):

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

## 4. XGBoost model

**Variant:** WOE + IV + RFE + complex XGBoost

| Item | Path |
|------|------|
| Training script | `scripts/train_xgboost_mmlu_pro_woe_iv_rfe_complex.py` |
| Notebook | `notebooks/train_xgboost_mmlu_pro_woe_iv_rfe_complex.ipynb` |
| Saved model | `models/xgboost_mmlu_pro_woe_iv_rfe_complex.json` |
| Preprocessing artifact | `models/xgboost_mmlu_pro_woe_iv_rfe_complex_preprocessing.json` |
| Metrics | `models/xgboost_mmlu_pro_woe_iv_rfe_complex_metrics.json` |

**Task:** Binary classification — predict whether a given model will get a given question wrong (`error` = 1) vs correct (`no_error` = 0).

---

## 5. Training pipeline — step by step

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

### Step 4 — Recursive Feature Elimination (RFE)

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

### Step 6 — Evaluation results

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

## 6. Final model feature set

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

---

## 7. Comparison with other project models

| Model | Dataset | Test rows | Test accuracy | Test ROC-AUC |
|-------|---------|-----------|---------------|--------------|
| **MMLU-Pro complex XGBoost** | `mmlu_pro_full_enriched.csv` | 84,569 | **0.730** | **0.808** |
| Combined complex XGBoost | `combined_full_enriched.csv` | 4,544 | 0.755 | 0.832 |

MMLU-Pro test set is ~19× larger. Accuracy is slightly lower because MMLU error/no_error is a near-random 50/50 split per model capability tier, while ReaLMistake/Mis-prompt in the combined set have different label dynamics.

---

## 8. Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/view_mmlu_pro_enriched.ipynb` | Explore enriched data: null counts, per-model samples, feature distributions |
| `notebooks/train_xgboost_mmlu_pro_woe_iv_rfe_complex.ipynb` | Full training pipeline: split → WOE → IV → RFE → XGBoost → metrics → save artifacts |

```bash
jupyter notebook notebooks/
```

---

## 9. End-to-end reproduction

```bash
source venv/bin/activate

python scripts/download_mmlu_pro.py
python scripts/build_mmlu_pro_dataset.py
python scripts/enrich_mmlu_pro_full.py
python scripts/train_xgboost_mmlu_pro_woe_iv_rfe_complex.py
```

Or run interactively via `notebooks/train_xgboost_mmlu_pro_woe_iv_rfe_complex.ipynb`.

---

## 10. Key takeaways

1. **563,787 model–question rows** from 47 LLMs on MMLU-Pro, labeled error vs no_error.
2. **32 engineered features** (model specs + question metrics) → WOE on 5 categoricals.
3. **IV + RFE** reduced to **15 features**; model identity and benchmark proxies are strongest, but RFE also kept question length/complexity and subject.
4. **Test performance:** 73.0% accuracy, 0.808 ROC-AUC on 84,569 held-out rows.
5. The model predicts **whether a specific LLM will fail on a specific question** using model metadata and lightweight question features — not the question text itself.

---

## References

- Wang et al., [MMLU-Pro](https://arxiv.org/abs/2406.01574) (NeurIPS 2024)
- Dataset: [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
- Eval results: [TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro)
