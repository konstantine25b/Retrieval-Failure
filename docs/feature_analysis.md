# Feature Engineering & Analysis — XGBoost WOE + IV + RFE Pipeline

This document describes every feature used in `notebooks/train_xgboost_woe_iv_rfe.ipynb`, how each one was created, and how it is transformed before modeling.

**Target:** predict `error` (1) vs `no_error` (0) from the ReaLMistake dataset (900 rows).

**Data source:** `data/realmistake_full_enriched.csv` (35 columns: 3 base + 23 model + 9 question-derived).

---

## End-to-end pipeline

```
realmistake_full.csv
        │
        ├─ enrich_realmistake_full.py  → 23 model-level features (joined by llm_model)
        │
        └─ enrich_question_features.py → 9 question-derived features (computed from question text)
                │
                ▼
        realmistake_full_enriched.csv
                │
                ▼
        train_xgboost_woe_iv_rfe.ipynb
                │
                ├─ Stratified split: 70% train / 15% val / 15% test (630 / 135 / 135)
                ├─ WOE encoding on 5 categoricals → 5 WOE columns
                ├─ Feature matrix: 21 numeric + 6 boolean + 5 WOE = 32 features
                ├─ Median imputation for missing numeric values
                ├─ IV filtering (threshold 0.02) → 10 features pass
                ├─ RFE with XGBoost (top 15 cap) → all 10 kept
                └─ Final XGBoost classifier on 10 selected features
```

---

## Feature creation (upstream enrichment)

Features are **not** computed inside the notebook. They are built in two scripts, run in order:

```bash
python scripts/enrich_realmistake_full.py
python scripts/enrich_question_features.py
```

### Model-level features (`scripts/enrich_realmistake_full.py`)

Each row has an `llm_model` value (`gpt-4-0613` or `meta-llama/Llama-2-70b-chat-hf`). The script looks up a static feature dictionary keyed by model name and appends 23 columns.

These values are **fixed per model** — they do not vary row-by-row within the same model. Architecture specs, API defaults, and external benchmark scores are hand-curated in `MODEL_FEATURES`.

### Question-derived features (`scripts/enrich_question_features.py`)

For each row, `extract_question_features(question, llm_model)` runs on the `question` column and returns 9 columns. Implementation uses:

- `textstat` — Flesch-Kincaid grade level
- `tiktoken` — GPT-4 token count (`cl100k_base` encoding)
- Regex patterns — few-shot detection, system instructions, negation
- Rule-based classifier — question category

---

## Notebook transformations

After loading the enriched CSV, the notebook applies three transformations before modeling.

### 1. WOE encoding (Weight of Evidence)

Applied to 5 categorical columns. Each category level is replaced with:

```
WOE = log( (P(event | level) + 0.5) / (P(non-event | level) + 0.5) )
```

- **Events** = rows where `target = 1` (error)
- **Non-events** = rows where `target = 0` (no_error)
- Laplace-style smoothing: `+0.5` in numerator, `+1.0` in denominator
- WOE maps are fit **only on the training set**, then applied to val/test
- Unknown levels at inference → filled with `0.0`

| Raw column | WOE column |
|------------|------------|
| `model_name` | `model_name_woe` |
| `positional_encoding_type` | `positional_encoding_type_woe` |
| `attention_type` | `attention_type_woe` |
| `tokenizer_type` | `tokenizer_type_woe` |
| `question_category` | `question_category_woe` |

**WOE values learned on training data:**

| Column | Level | WOE |
|--------|-------|-----|
| `model_name` | `gpt-4-0613` | −0.483 |
| `model_name` | `llama-2-70b-chat-hf` | +0.528 |
| `positional_encoding_type` | `learned_absolute` | −0.483 |
| `positional_encoding_type` | `RoPE` | +0.528 |
| `attention_type` | `MHA` | −0.483 |
| `attention_type` | `GQA` | +0.528 |
| `tokenizer_type` | `cl100k_BPE` | −0.483 |
| `tokenizer_type` | `sentencepiece_BPE` | +0.528 |
| `question_category` | `Coding` | −0.014 |
| `question_category` | `Creative` | +0.111 |
| `question_category` | `Fact Retrieval` | +0.064 |
| `question_category` | `Reasoning` | −0.042 |

Positive WOE → that level is associated with higher error rate than average. The four model-architecture WOE columns share the same two values because they are perfectly correlated with `llm_model` in this dataset (only two models).

### 2. Feature matrix assembly

```python
numeric  = frame[NUMERIC_COLUMNS]           # 21 columns, coerced to float
boolean  = frame[BOOLEAN_COLUMNS].astype(int)  # 6 columns → 0/1
woe_cols = [f"{col}_woe" for col in CATEGORICAL_COLUMNS]  # 5 columns
X = concat(numeric, boolean, woe_cols)      # 32 features total
```

### 3. Missing value imputation

Training-set medians are computed and used to fill NaN in train, val, and test. This mainly affects model-specific fields that are not applicable to both models (e.g. `top_k` is NA for GPT-4, `frequency_penalty` is NA for Llama).

---

## All 32 features

### A. Model architecture & capacity (numeric)

| Feature | Meaning | How it was created | GPT-4-0613 | Llama-2-70b | Train median | IV | Selected |
|---------|---------|-------------------|------------|-------------|--------------|-----|----------|
| `context_window_tokens` | Maximum input context the model can accept (tokens) | Static lookup in `MODEL_FEATURES` by `llm_model` | 8192 | 4096 | 4096 | 0.000 | No |
| `max_output_tokens` | Maximum tokens the model can generate in one response | Static lookup | 4096 | 2048 | 2048 | 0.000 | No |
| `vocab_size` | Tokenizer vocabulary size | Static lookup | 100277 | 32000 | 32000 | 0.000 | No |
| `knowledge_cutoff_year` | Last year of training data the model was exposed to | Static lookup | 2021 | 2022 | 2022 | 0.000 | No |

**Note:** IV = 0 because each value is constant within a model, and the dataset has only two models. These columns are redundant with model identity.

---

### B. Model architecture (categorical → WOE)

| Feature | Meaning | How it was created | GPT-4-0613 | Llama-2-70b | IV (on raw) | WOE col IV | Selected |
|---------|---------|-------------------|------------|-------------|-------------|------------|----------|
| `model_name` | Short model identifier | Mapped from `llm_model` | `gpt-4-0613` | `llama-2-70b-chat-hf` | — | 0.252 | Yes (`model_name_woe`) |
| `positional_encoding_type` | How position information is injected into embeddings | Static lookup | `learned_absolute` | `RoPE` | — | 0.252 | Yes (`positional_encoding_type_woe`) |
| `attention_type` | Attention mechanism variant | Static lookup | `MHA` (multi-head) | `GQA` (grouped-query) | — | 0.252 | Yes (`attention_type_woe`) |
| `tokenizer_type` | Tokenization scheme | Static lookup | `cl100k_BPE` | `sentencepiece_BPE` | — | 0.252 | Yes (`tokenizer_type_woe`) |

These four WOE columns are **perfectly collinear** with each other and with `is_open_source` / `multilingual_support` in this two-model dataset. IV is identical (0.252) because they partition the data the same way.

---

### C. Model licensing & capabilities (boolean)

| Feature | Meaning | How it was created | GPT-4-0613 | Llama-2-70b | Train median | IV | Selected |
|---------|---------|-------------------|------------|-------------|--------------|-----|----------|
| `is_open_source` | Whether the model weights are publicly available | Static lookup → `False` / `True` → cast to 0/1 | 0 | 1 | 1.0 | 0.252 | Yes |
| `multilingual_support` | Whether the model supports multiple languages | Static lookup | 1 (True) | 0 (False) | 0.0 | 0.252 | Yes |

Both are anti-correlated with each other in this dataset and carry the same information as model identity.

---

### D. Generation hyperparameters (numeric)

| Feature | Meaning | How it was created | GPT-4-0613 | Llama-2-70b | Train median | IV | Selected |
|---------|---------|-------------------|------------|-------------|--------------|-----|----------|
| `temperature` | Sampling temperature (higher = more random) | Static lookup (API defaults, not per-row inference settings) | 1.0 | 0.6 | 0.6 | 0.000 | No |
| `top_p` | Nucleus sampling cutoff | Static lookup | 1.0 | 0.9 | 0.9 | 0.000 | No |
| `top_k` | Top-k sampling limit | Static lookup; NA for GPT-4 | NA → imputed | 50 | 50.0 | 0.000 | No |
| `repetition_penalty` | Penalty for repeating tokens (Llama-specific) | Static lookup; NA for GPT-4 | NA → imputed | 1.2 | 1.2 | 0.000 | No |
| `frequency_penalty` | OpenAI frequency penalty | Static lookup; NA for Llama | 0.0 | NA → imputed | 0.0 | 0.000 | No |
| `presence_penalty` | OpenAI presence penalty | Static lookup; NA for Llama | 0.0 | NA → imputed | 0.0 | 0.000 | No |
| `max_tokens_requested` | Max tokens requested in API call | Static lookup | 4096 | 1024 | 1024 | 0.000 | No |
| `stop_sequences_count` | Number of stop sequences configured | Static lookup | 0 | 1 | 1.0 | 0.000 | No |

**Note:** ReaLMistake used greedy decoding (temperature 0.0) at inference time. These columns reflect documented API defaults, not actual per-row generation settings. IV = 0 for all because values are constant per model.

---

### E. External benchmark scores (numeric)

| Feature | Meaning | How it was created | GPT-4-0613 | Llama-2-70b | Train median | IV | Selected |
|---------|---------|-------------------|------------|-------------|--------------|-----|----------|
| `galileo_qa_no_rag` | Galileo QA benchmark score without RAG | Static lookup (external benchmark) | 0.77 | 0.65 | 0.65 | 0.000 | No |
| `galileo_qa_with_rag` | Galileo QA benchmark score with RAG | Static lookup | 0.76 | 0.68 | 0.68 | 0.000 | No |
| `galileo_longform` | Galileo long-form generation score | Static lookup | 0.83 | 0.82 | 0.82 | 0.000 | No |
| `crag_hallucination_rate` | CRAG benchmark hallucination rate (lower is better) | Static lookup | 0.135 | 0.287 | 0.287 | 0.000 | No |
| `crag_accuracy` | CRAG benchmark accuracy | Static lookup | 0.335 | 0.223 | 0.223 | 0.000 | No |

Benchmark scores differ between models but are constant within each model, so IV = 0 under the same two-model constraint.

---

### F. Question length & complexity (numeric)

| Feature | Meaning | How it was created | Train median | IV | Selected |
|---------|---------|-------------------|--------------|-----|----------|
| `question_length_words` | Number of whitespace-separated words in the prompt | `len(question.split())` in `extract_question_features()` | 271.5 | **0.344** | **Yes** |
| `question_length_chars` | Total character count of the prompt | `len(question)` | 1729.0 | **0.453** | **Yes** |
| `question_complexity_score` | Estimated U.S. school grade level needed to read the prompt | `textstat.flesch_kincaid_grade(question)`, rounded to 2 decimals | 10.23 | **0.064** | **Yes** |
| `context_token_count` | Estimated token length of the prompt for the target model | GPT-4: `len(tiktoken cl100k_base.encode(question))`; Llama: `int(word_count × 1.25)` | 335.0 | **0.372** | **Yes** |

These are the **strongest predictors** in the dataset. They vary row-by-row (different prompts have different lengths) and capture how demanding the input is. Longer, more complex prompts tend to correlate with higher error rates.

**IV computation for numerics:** values are binned into 5 quantile groups (`pd.qcut`, `NUMERIC_IV_BINS=5`), then IV is computed on the bins vs target.

---

### G. Question content flags (boolean)

| Feature | Meaning | How it was created | Train median | IV | Selected |
|---------|---------|-------------------|--------------|-----|----------|
| `has_few_shot_examples` | Prompt contains few-shot examples like "Example 1:" | Regex: `(?i)example\s+\d+\s*:` | 0.0 | 0.000 | No |
| `prompt_contains_system_instructions` | Prompt contains explicit system/instruction block | Regex: `(?i)(?:^|\n)\s*system\s*:` or `(?:^|\n)\s*instructions?\s*:` | 0.0 | 0.000 | No |
| `is_ambiguous` | Prompt is very short (< 5 words), potentially underspecified | `word_count < 5` | 0.0 | 0.000 | No |
| `contains_negation` | Prompt contains negation words | Regex on: `not`, `never`, `no`, `cannot`, `can't`, `won't`, `don't`, `doesn't`, `isn't`, `aren't`, `shouldn't`, `wouldn't`, `couldn't`, `mustn't`, `hardly`, `barely`, etc. | 1.0 | 0.000 | No |

All four flags had IV = 0 on the training set. `contains_negation` is almost always True (median = 1.0) because ReaLMistake prompts frequently include negation in instructions. `has_few_shot_examples` and `prompt_contains_system_instructions` are almost always False.

---

### H. Question category (categorical → WOE)

| Feature | Meaning | How it was created | IV (raw) | WOE col IV | Selected |
|---------|---------|-------------------|----------|------------|----------|
| `question_category` | High-level task type of the prompt | Rule-based classifier in `classify_question_category()` | 0.004 | 0.004 | **No** |

**Classification rules** (first match wins):

| Category | Trigger condition |
|----------|-------------------|
| `Coding` | Contains `def`, `class`, `import`, `function`, `python`, `javascript`, `sql`, `algorithm`, or `code` |
| `Creative` | Contains `story`, `poem`, `creative`, `fiction`, `narrative`, or `write a song` |
| `Fact Retrieval` | Contains `claim and evidence`, `wikipedia article`, or `fact verification` |
| `Reasoning` | Contains `math word problem`, `generate a math`, `unanswerable`, or `answer the following question`; also the default fallback |

**Distribution:** most rows are `Reasoning` (math word problems dominate the dataset). IV = 0.004, below the 0.02 threshold — dropped during IV filtering.

---

## Feature selection results

### Information Value (IV) filtering

IV measures how well a feature separates errors from non-errors:

```
IV = Σ (P(event|bin) − P(non-event|bin)) × WOE(bin)
```

| IV range | Interpretation |
|----------|----------------|
| < 0.02 | Weak — not useful for prediction |
| 0.02 – 0.1 | Medium |
| 0.1 – 0.3 | Strong |
| > 0.3 | Suspicious (possible overfitting) |

**Threshold:** `IV_THRESHOLD = 0.02`

| Rank | Feature | IV | Passes IV? |
|------|---------|-----|------------|
| 1 | `question_length_chars` | 0.453 | Yes |
| 2 | `context_token_count` | 0.372 | Yes |
| 3 | `question_length_words` | 0.344 | Yes |
| 4 | `tokenizer_type_woe` | 0.252 | Yes |
| 5 | `attention_type_woe` | 0.252 | Yes |
| 6 | `positional_encoding_type_woe` | 0.252 | Yes |
| 7 | `model_name_woe` | 0.252 | Yes |
| 8 | `multilingual_support` | 0.252 | Yes |
| 9 | `is_open_source` | 0.252 | Yes |
| 10 | `question_complexity_score` | 0.064 | Yes |
| 11 | `question_category_woe` | 0.004 | No |
| 12–32 | All remaining features | 0.000 | No |

**10 of 32** features passed IV filtering.

### Recursive Feature Elimination (RFE)

- **Estimator:** XGBoost (100 trees, max_depth=3) wrapped in sklearn `RFE`
- **Target:** select up to `RFE_N_FEATURES = 15` features from the IV-passing set
- **Result:** all 10 IV-passing features received RFE rank 1 (all kept)

Since only 10 features passed IV and the RFE cap was 15, RFE had no effect — the final model uses the same 10 features.

### Final 10 features used by XGBoost

| # | Feature | Group | Why it survived |
|---|---------|-------|-----------------|
| 1 | `question_length_chars` | Question numeric | Highest IV (0.453); varies per row |
| 2 | `context_token_count` | Question numeric | IV 0.372; token-aware prompt length |
| 3 | `question_length_words` | Question numeric | IV 0.344; word-count proxy for prompt size |
| 4 | `question_complexity_score` | Question numeric | IV 0.064; readability / difficulty signal |
| 5 | `is_open_source` | Model boolean | IV 0.252; proxies for Llama vs GPT-4 |
| 6 | `multilingual_support` | Model boolean | IV 0.252; anti-correlated model proxy |
| 7 | `model_name_woe` | Model WOE | IV 0.252; WOE-encoded model identity |
| 8 | `positional_encoding_type_woe` | Model WOE | IV 0.252; architecture proxy |
| 9 | `attention_type_woe` | Model WOE | IV 0.252; architecture proxy |
| 10 | `tokenizer_type_woe` | Model WOE | IV 0.252; architecture proxy |

---

## Key observations

### What actually drives predictions

1. **Question length features** (`question_length_chars`, `context_token_count`, `question_length_words`) are the only features that vary meaningfully across rows within the same model. They carry the most independent predictive signal.

2. **Model identity features** (`is_open_source`, `multilingual_support`, and the four WOE columns) are six different encodings of the same binary split (GPT-4 vs Llama-2). They all have IV = 0.252 and are perfectly collinear. XGBoost can use any one of them; having six is redundant but harmless for tree models.

3. **22 features had IV = 0** — every model-level numeric/benchmark/hyperparameter column, plus all four question boolean flags. These are either constant per model or nearly constant across the dataset.

### Limitations

- Only **two models** exist in the data. Any feature that is constant within a model cannot have IV > 0, regardless of how meaningful it might be in a multi-model setting.
- **Generation hyperparameters** (temperature, top_p, etc.) are API defaults, not the actual greedy-decoding settings used in ReaLMistake evaluation.
- **`question_category`** is dominated by `Reasoning` (~math word problems), giving it almost no discriminative power.
- **Train vs test gap** (90% train accuracy vs 76% test) suggests overfitting, partly driven by redundant model-proxy features.

---

## Source files

| File | Role |
|------|------|
| `scripts/enrich_realmistake_full.py` | Adds 23 model-level features from `MODEL_FEATURES` dict |
| `scripts/enrich_question_features.py` | Adds 9 question-derived features via regex, textstat, tiktoken |
| `notebooks/train_xgboost_woe_iv_rfe.ipynb` | WOE encoding, IV filtering, RFE, XGBoost training |
| `scripts/train_xgboost_woe_iv_rfe.py` | Script equivalent of the notebook |
| `models/xgboost_woe_iv_rfe_preprocessing.json` | Saved WOE maps, medians, IV scores, selected features |
| `models/xgboost_woe_iv_rfe_metrics.json` | Train/val/test evaluation metrics |
