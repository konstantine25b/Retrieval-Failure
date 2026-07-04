# MMLU-Pro XGBoost — No Category Variant

Companion experiment to the full MMLU-Pro notebook.  
**Key constraint:** only three raw columns from the dataset are allowed — `question`, `llm_model`, and `error`. No pre-computed metadata (architecture specs, benchmark scores, question category) is used. Every feature is derived from scratch.

This lets us measure how much of the retrieval-failure signal lives purely in the question text and the model identity, independently of any domain label.

---

## Notebook

`notebooks/train_xgboost_mmlu_pro_no_category.ipynb`

Runs the full pipeline end-to-end:

1. Load 3 columns from `data/mmlu_pro_full_enriched.csv`
2. Engineer 41 features from question text + model name
3. Save enriched dataset → `data/mmlu_pro_question_features.csv`
4. Stratified 70 / 15 / 15 split
5. WOE-encode 2 categorical columns (fitted on train only)
6. IV filter (threshold = 0, keeps all with any signal)
7. RFE → top 25 features
8. Complex XGBoost (1 000 trees, depth = 7, lr = 0.02, early stopping = 50)
9. Evaluate on held-out test set
10. Save model + artifacts → `models/xgboost_mmlu_pro_no_category.*`

---

## Output dataset

`data/mmlu_pro_question_features.csv`

Contains the original three columns plus all engineered features below.

---

## Feature dictionary

### Categorical → WOE encoded

| Feature | Type | Description |
|---|---|---|
| `llm_model` → `llm_model_woe` | categorical | Which LLM was queried. WOE value reflects the model's error rate relative to the global base rate. High WOE = higher error tendency for that model. |
| `q_expected_answer_type` → `q_expected_answer_type_woe` | categorical | Inferred answer type from the question stem: `numeric`, `true_false`, `formula`, `named_entity`, `concept`, or `other`. Detected with rule-based keyword matching on the stem (text before answer choices). |

---

### Length & size (8 numeric features)

| Feature | Description |
|---|---|
| `q_word_count` | Total number of words in the full question string (including answer choices). |
| `q_char_count` | Total non-whitespace character count. |
| `q_sentence_count` | Number of sentences detected by splitting on `.`, `?`, `!`. Clipped to minimum 1. |
| `q_newline_count` | Number of `\n` characters in the raw text. More newlines typically indicate a structured multi-choice layout. |
| `q_unique_word_count` | Count of distinct (lowercased) word tokens. |
| `q_avg_word_length` | Mean length of word tokens in characters. Longer average = more technical or domain-specific vocabulary. |
| `q_type_token_ratio` | Unique words / total words. Measures lexical diversity; low ratio = repetitive phrasing (common in templated questions). |
| `q_avg_sentence_length` | Mean number of words per sentence (`q_word_count / q_sentence_count`). |

---

### Answer-choice structure (4 numeric features)

These features are computed by splitting the question into the *stem* (text before the first choice line) and *choice lines* (lines matching `A.` / `A)` … `J.` / `J)`).

| Feature | Description |
|---|---|
| `q_num_choices` | Number of answer option lines detected. Typically 4 (A–D) for standard MMLU but up to 10 (A–J) for MMLU-Pro. Questions with more options tend to be harder. |
| `q_stem_word_count` | Word count of the stem only (before first choice line). Longer stems = more context, potentially more ambiguity. |
| `q_stem_char_count` | Character count of the stem (non-whitespace). |
| `q_avg_choice_length_chars` | Mean character length of each answer choice line. Longer choices suggest more nuanced or verbose options. |

---

### Numeric & math content (6 features)

| Feature | Description |
|---|---|
| `q_has_numbers` | Binary. 1 if the question contains any digit character. |
| `q_digit_ratio` | Digit characters / total characters. Higher ratio = more numerical content. |
| `q_has_math_operators` | Binary. 1 if any of `+ - * / = ≥ ≤ ∑ √ % ∫ ∂ ∇ × ÷` appears in the text. |
| `q_math_operator_count` | Count of math operator characters. |
| `q_bracket_count` | Count of `(`, `)`, `[`, `]`, `{`, `}`, `⟨`, `⟩`. High counts indicate mathematical expressions or structured notation. |
| `q_has_scientific_notation` | Binary. 1 if the question matches patterns like `3.2e+5`, `1E-10`, or `× 10`. |

---

### Text style (5 numeric features)

| Feature | Description |
|---|---|
| `q_uppercase_ratio` | Uppercase letters / total letter characters. Questions with many acronyms or emphasized terms score higher. |
| `q_punctuation_count` | Total count of punctuation characters (using `string.punctuation`). |
| `q_comma_count` | Number of commas. High comma count suggests list-heavy or complex sentence structure. |
| `q_question_mark_count` | Number of `?` characters. Multi-part questions have more than 1. |
| `q_long_word_ratio` | Fraction of words with more than 6 characters. Proxy for domain-specific vocabulary density. |

---

### Semantic flags (10 binary features)

| Feature | Description |
|---|---|
| `q_has_negation` | 1 if the text contains `not`, `except`, `never`, `cannot`, or `n't`. Negation in questions is a known source of LLM error. |
| `q_has_none_all_above` | 1 if the text contains "none of the above" or "all of the above". These options are distractors that increase answer difficulty. |
| `q_is_which_following` | 1 if the stem contains "which of the following". Very common MMLU phrasing. |
| `q_has_code_pattern` | 1 if the text contains backtick, `def `, `class `, `import `, `for `, `while `, `function`, or `=>`. Signals a programming question. |
| `q_starts_with_what` | 1 if the stem (lowercased, stripped) starts with "what". |
| `q_starts_with_how` | 1 if the stem starts with "how". How-questions often require procedural reasoning. |
| `q_starts_with_why` | 1 if the stem starts with "why". Why-questions require causal reasoning. |
| `q_starts_with_which` | 1 if the stem starts with "which". Often selection from options. |
| `q_has_year_mention` | 1 if any 4-digit year in range 1900–2029 appears in the question. |
| `q_has_rare_entity` | 1 if two or more consecutive Title-Case words appear (e.g. "Albert Einstein", "United States"). Proxy for named entities that may be rare in training data. |

---

### Density (2 numeric features)

| Feature | Description |
|---|---|
| `q_stopword_ratio` | Stopwords / total words, using a hardcoded 90-word English stopword list (no NLTK). Low ratio = information-dense text; high ratio = conversational/verbose phrasing. |
| `q_dependency_depth_proxy` | Count of subordinating conjunctions and relative pronouns (`that`, `which`, `because`, `when`, `while`, `although`, `unless`, `since`, `after`, `before`, `until`, `whether`, `if`). Proxy for syntactic depth / sentence complexity. |

---

### Temporal / recency (3 features)

| Feature | Description |
|---|---|
| `q_max_year_mentioned` | The highest 4-digit year (1900–2029) found in the question. 0 if no year mentioned. Questions about recent events are more likely to fall outside a model's knowledge cutoff. |
| `q_recency_gap` | `2024 − q_max_year_mentioned` (0 if no year mentioned). Larger gap = older content, smaller gap = more recent content that the model may not know well. Inspired by `knowledge_cutoff_year` from the full dataset but derived purely from text. |

---

### Named-entity proxies (2 features — included in binary flags table above)

| Feature | Description |
|---|---|
| `q_num_named_entities` | Count of Title-Case words (e.g. `Napoleon`, `Mitochondria`) that do not appear as the first word of a sentence — i.e., capitalization is semantic, not grammatical. Proxy for proper-noun density. |
| `q_has_rare_entity` | (see binary flags table above) |

---

### Structural / reasoning proxies (2 features)

| Feature | Description |
|---|---|
| `q_is_leading_question` | Binary. 1 if the stem starts with a presupposition-implying phrase: `why did`, `what caused`, `how did`, `when did`, `who caused`, `what led`, `what made`, `why was/were/is/are`. These questions assume an event occurred and test causal reasoning. |
| `q_dependency_depth_proxy` | (see density table above) |

---

## Pipeline details

### WOE encoding

Weight of Evidence is fitted **on the training set only** to prevent data leakage. The formula per category level `v` is:

```
WOE(v) = ln( P(error | v) / P(no_error | v) )
```

with Laplace smoothing (+0.5 events, +1 total) to handle rare levels.  
Unseen levels at inference time are mapped to WOE = 0.0 (global average).

### Information Value (IV) filtering

IV for each feature is computed on the training set using 5-quantile bins for numeric features and direct grouping for boolean and WOE features. Threshold is set to 0 (all features with any discriminating power pass). The full ranked IV table is displayed in the notebook.

### RFE

Recursive Feature Elimination uses a lightweight XGBoost (`n_estimators=100`, `max_depth=4`) to rank features. Eliminates one feature per step until exactly **25** remain. These 25 features are used for the final model.

### XGBoost hyperparameters

| Parameter | Value |
|---|---|
| `n_estimators` | 1 000 |
| `max_depth` | 7 |
| `learning_rate` | 0.02 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `early_stopping_rounds` | 50 |
| `eval_metric` | logloss |

Early stopping is evaluated on the validation set.

---

## Comparison with full model

| Aspect | Full model | No-category model |
|---|---|---|
| Notebook | `train_xgboost_mmlu_pro_woe_iv_rfe_complex.ipynb` | `train_xgboost_mmlu_pro_no_category.ipynb` |
| Input columns used | All 35 columns | 3 columns only |
| Feature count (pre-RFE) | 32 | 41 |
| RFE target | 15 | 25 |
| Includes `question_category` | Yes (WOE) | No |
| Includes model architecture metadata | Yes | No |
| Includes benchmark scores | Yes | No |
| Features from question text | 5 (length/negation) | 30+ (full text analysis) |

The no-category model is a **lower bound** on model performance achievable without domain labels, and a **diagnostic tool** to quantify how much information the question structure alone contains about retrieval failure.

---

---

## Run results (Jul 4 2026)

### Data

| Split | Rows |
|---|---|
| Train | 394,650 |
| Validation | 84,568 |
| Test | 84,569 |
| **Total** | **563,787** |

Classes are near-perfectly balanced: 50.15% error / 49.85% no_error.

---

### Information Value (IV) — all 41 features ranked

| Rank | Feature | IV | Interpretation |
|---|---|---|---|
| 1 | `llm_model_woe` | **0.5669** | Strong — model identity dominates |
| 2 | `q_stem_word_count` | 0.1431 | Medium — proxies question domain/difficulty |
| 3 | `q_stem_char_count` | 0.1236 | Medium — highly correlated with above |
| 4 | `q_punctuation_count` | 0.1109 | Medium |
| 5 | `q_sentence_count` | 0.0766 | Medium |
| 6 | `q_has_numbers` | 0.0693 | Weak–medium |
| 7 | `q_digit_ratio` | 0.0666 | Weak–medium |
| 8 | `q_avg_word_length` | 0.0608 | Weak–medium |
| 9 | `q_long_word_ratio` | 0.0541 | Weak–medium |
| 10 | `q_math_operator_count` | 0.0453 | Weak–medium |
| 11 | `q_expected_answer_type_woe` | 0.0424 | Weak–medium |
| 12 | `q_bracket_count` | 0.0353 | Weak |
| 13 | `q_word_count` | 0.0330 | Weak |
| 14 | `q_has_math_operators` | 0.0295 | Weak |
| 15 | `q_starts_with_what` | 0.0257 | Weak |
| 16 | `q_comma_count` | 0.0245 | Weak |
| 17 | `q_stopword_ratio` | 0.0234 | Weak |
| 18 | `q_unique_word_count` | 0.0220 | Weak |
| 19 | `q_type_token_ratio` | 0.0204 | Weak |
| 20 | `q_has_scientific_notation` | 0.0116 | Very weak |
| 21 | `q_char_count` | 0.0111 | Very weak |
| 22 | `q_avg_choice_length_chars` | 0.0093 | Very weak |
| 23 | `q_has_code_pattern` | 0.0084 | Very weak |
| 24 | `q_num_named_entities` | 0.0066 | Very weak |
| 25 | `q_uppercase_ratio` | 0.0062 | Very weak |
| 26 | `q_starts_with_why` | 0.0061 | Very weak |
| 27 | `q_newline_count` | 0.0057 | Very weak |
| 28 | `q_starts_with_how` | 0.0054 | Very weak |
| 29 | `q_avg_sentence_length` | 0.0054 | Very weak |
| 30 | `q_is_leading_question` | 0.0048 | Very weak |
| 31 | `q_dependency_depth_proxy` | 0.0042 | Very weak |
| 32 | `q_has_negation` | 0.0036 | Very weak |
| 33 | `q_starts_with_which` | 0.0022 | Very weak |
| 34 | `q_has_year_mention` | 0.0007 | Negligible |
| 35 | `q_recency_gap` | 0.0007 | Negligible |
| 36 | `q_is_which_following` | 0.0002 | Negligible |
| 37 | `q_num_choices` | 0.0002 | Negligible |
| 38 | `q_has_rare_entity` | 0.0001 | Negligible |
| 39 | `q_has_none_all_above` | 0.0001 | Negligible |
| 40 | `q_question_mark_count` | 0.0001 | Negligible |
| 41 | `q_max_year_mentioned` | 0.0000 | None |

IV guidance: < 0.02 = weak, 0.02–0.1 = medium, 0.1–0.3 = strong, > 0.3 = very strong.

---

### RFE — selected vs eliminated

All 41 features passed IV threshold (= 0). RFE then selected the top 25.

**Selected (25):**

| # | Feature | RFE rank |
|---|---|---|
| 1 | `llm_model_woe` | 1 |
| 2 | `q_stem_word_count` | 1 |
| 3 | `q_stem_char_count` | 1 |
| 4 | `q_punctuation_count` | 1 |
| 5 | `q_sentence_count` | 1 |
| 6 | `q_digit_ratio` | 1 |
| 7 | `q_avg_word_length` | 1 |
| 8 | `q_long_word_ratio` | 1 |
| 9 | `q_math_operator_count` | 1 |
| 10 | `q_expected_answer_type_woe` | 1 |
| 11 | `q_bracket_count` | 1 |
| 12 | `q_word_count` | 1 |
| 13 | `q_starts_with_what` | 1 |
| 14 | `q_stopword_ratio` | 1 |
| 15 | `q_unique_word_count` | 1 |
| 16 | `q_type_token_ratio` | 1 |
| 17 | `q_char_count` | 1 |
| 18 | `q_avg_choice_length_chars` | 1 |
| 19 | `q_newline_count` | 1 |
| 20 | `q_has_negation` | 1 |
| 21 | `q_starts_with_which` | 1 |
| 22 | `q_is_which_following` | 1 |
| 23 | `q_num_choices` | 1 |
| 24 | `q_has_rare_entity` | 1 |
| 25 | `q_question_mark_count` | 1 |

**Eliminated (16):**

| Feature | RFE rank |
|---|---|
| `q_avg_sentence_length` | 2 |
| `q_has_none_all_above` | 3 |
| `q_starts_with_how` | 4 |
| `q_starts_with_why` | 5 |
| `q_comma_count` | 6 |
| `q_num_named_entities` | 7 |
| `q_uppercase_ratio` | 8 |
| `q_dependency_depth_proxy` | 9 |
| `q_recency_gap` | 10 |
| `q_is_leading_question` | 11 |
| `q_max_year_mentioned` | 12 |
| `q_has_scientific_notation` | 13 |
| `q_has_numbers` | 14 |
| `q_has_math_operators` | 15 |
| `q_has_year_mention` | 16 |
| `q_has_code_pattern` | 17 |

Notable: `q_has_numbers` / `q_has_math_operators` (binary flags) were dropped in favour of their continuous counterparts `q_digit_ratio` / `q_math_operator_count` — the count/ratio forms carry more gradient signal for trees.

---

### XGBoost training curve

| Iteration | Val logloss |
|---|---|
| 0 | 0.68944 |
| 100 | 0.58019 |
| 200 | 0.56168 |
| 300 | 0.54926 |
| 400 | 0.53852 |
| 500 | 0.52925 |
| 600 | 0.52097 |
| 700 | 0.51348 |
| 800 | 0.50655 |
| 900 | 0.50049 |
| 999 | 0.49445 |

**Best iteration: 999** — early stopping did not trigger. The model was still improving at the 1000-tree limit, meaning the logloss had not converged. Increasing `n_estimators` to 1500–2000 could squeeze out additional accuracy.

---

### Test set metrics

| Metric | Value |
|---|---|
| **Accuracy** | **0.7663** |
| **F1** | **0.7670** |
| **Precision** | **0.7674** |
| **Recall** | **0.7666** |
| **ROC-AUC** | **0.8457** |

```
              precision    recall  f1-score   support

    no_error       0.77      0.77      0.77     42,142
       error       0.77      0.77      0.77     42,427

    accuracy                           0.77     84,569
   macro avg       0.77      0.77      0.77     84,569
weighted avg       0.77      0.77      0.77     84,569
```

**Confusion matrix:**

```
              Predicted no_error   Predicted error
Actual no_error      32,281             9,861
Actual error          9,901            32,526
```

- False positive rate (no_error → error): 23.4%
- False negative rate (error → no_error): 23.3%
- Both classes predicted with equal accuracy — model is well-calibrated.

---

### Key observations from this run

1. **Model identity dominates (IV = 0.567).** The `llm_model_woe` feature alone captures most of the signal. Different LLMs have very different overall error rates (e.g. Llama-2-7b errors far more often than DeepSeek-Coder-V2). The model learns this per-model tendency as its strongest signal.

2. **`q_stem_word_count` / `q_stem_char_count` are implicit domain proxies (IV ≈ 0.12–0.14).** MMLU-Pro domains have structurally different question lengths: medicine/law use long case descriptions, while math/CS use short symbolic stems. Without any explicit category label, the model discovered this correlation.

3. **Math content features cluster together.** `q_math_operator_count`, `q_bracket_count`, `q_digit_ratio`, `q_has_scientific_notation`, and `q_has_math_operators` all carry related signal. RFE kept the count/ratio forms and dropped the binary flags.

4. **Early stopping did not trigger** — the model was still learning at iteration 999. Increasing `n_estimators` to 1500+ is likely to improve accuracy by ~0.5–1%.

5. **Compared to the full model:** the full notebook (`train_xgboost_mmlu_pro_woe_iv_rfe_complex.ipynb`) uses 32 features including explicit model architecture metadata, hallucination benchmark scores, and `question_category`. The gap between that model and this one (76.63%) quantifies exactly how much those external metadata columns contribute beyond what raw question text already encodes.

---

## How to run

```bash
# Activate the project virtual environment
source venv/bin/activate

# Open Jupyter
jupyter lab notebooks/train_xgboost_mmlu_pro_no_category.ipynb
```

Run all cells top-to-bottom. Feature engineering takes approximately 3–6 minutes for 560 k rows on a modern laptop. The enriched CSV is saved to `data/mmlu_pro_question_features.csv` before the split, so you can reuse it without re-running the engineering step.
