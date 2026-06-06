# Retrieval Failure

Analysis and modeling pipeline for the [ReaLMistake](https://arxiv.org/abs/2404.03602) benchmark: LLM error detection on 900 expert-annotated examples from GPT-4 and Llama 2 70B across three tasks.

## Project structure

```
.
├── data/
│   ├── realmistake/                  # Source JSONL (3 tasks × 2 models)
│   ├── realmistake_full.csv          # Base tabular export (900 rows)
│   └── realmistake_full_enriched.csv # Full feature set (900 rows × 35 cols)
├── scripts/
│   ├── download_realmistake.py       # Download dataset from Hugging Face or zip
│   ├── build_full_dataset.py         # JSONL → realmistake_full.csv
│   ├── enrich_realmistake_full.py      # Add model-level features
│   ├── enrich_question_features.py   # Add question-based features
│   └── train_xgboost_woe.py          # WOE + XGBoost training & evaluation
├── notebooks/
│   ├── analyze_realmistake_full.ipynb
│   ├── view_realmistake_enriched.ipynb
│   └── train_xgboost_woe.ipynb
├── models/
│   ├── xgboost_woe.json              # Trained classifier
│   ├── preprocessing.json            # WOE maps & imputation values
│   └── metrics.json                  # Train / val / test scores
└── requirements.txt
```

## Setup

```bash
cd "Retrival Failure"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 1. Download data

Fetches ReaLMistake from Hugging Face (`ryokamoi/realmistake`) or falls back to the official password-protected zip.

```bash
python scripts/download_realmistake.py
```

Output: `data/realmistake/` with three task folders:

- `math_word_problem_generation`
- `finegrained_fact_verification`
- `answerability_classification`

Each folder contains JSONL files for `gpt-4-0613` and `meta-llama/Llama-2-70b-chat-hf`.

### Source JSONL schema

Each line includes:

| Field | Description |
|-------|-------------|
| `input` | Task prompt sent to the model |
| `llm_response` | Model output |
| `error_label` | `error` or `no_error` |
| `human_explanation` | Expert rationale |
| `error_categories` | e.g. Instruction-Following, Context-Faithfulness |
| `metadata` | `llm_response_model`, task name, difficulty, id |

Inference hyperparameters (temperature, top_p, etc.) are **not** stored in the data — only model names. Generation settings are documented in ReaLMistake Appendix H (temperature 0.0, greedy decoding, max output tokens).

## 2. Build base CSV

```bash
python scripts/build_full_dataset.py
```

Creates `data/realmistake_full.csv` (900 rows):

| Column | Description |
|--------|-------------|
| `question` | Task prompt (from `input`) |
| `llm_model` | `gpt-4-0613` or `meta-llama/Llama-2-70b-chat-hf` |
| `error` | `error` (649) or `no_error` (251) |

## 3. Feature enrichment

Run both scripts in order:

```bash
python scripts/enrich_realmistake_full.py
python scripts/enrich_question_features.py
```

Output: `data/realmistake_full_enriched.csv` — **900 rows × 35 columns**.

### Model-level features (23 columns)

Joined by `llm_model`. Static architecture, API defaults, and external benchmark scores.

| Column | GPT-4-0613 | Llama-2-70b-chat-hf |
|--------|------------|---------------------|
| `model_name` | gpt-4-0613 | llama-2-70b-chat-hf |
| `context_window_tokens` | 8192 | 4096 |
| `max_output_tokens` | 4096 | 2048 |
| `vocab_size` | 100277 | 32000 |
| `positional_encoding_type` | learned_absolute | RoPE |
| `attention_type` | MHA | GQA |
| `tokenizer_type` | cl100k_BPE | sentencepiece_BPE |
| `is_open_source` | False | True |
| `knowledge_cutoff_year` | 2021 | 2022 |
| `multilingual_support` | True | False |
| `temperature` | 1.0 | 0.6 |
| `top_p` | 1.0 | 0.9 |
| `top_k` | — | 50 |
| `repetition_penalty` | — | 1.2 |
| `frequency_penalty` | 0.0 | — |
| `presence_penalty` | 0.0 | — |
| `max_tokens_requested` | 4096 | 1024 |
| `stop_sequences_count` | 0 | 1 |
| `galileo_qa_no_rag` | 0.77 | 0.65 |
| `galileo_qa_with_rag` | 0.76 | 0.68 |
| `galileo_longform` | 0.83 | 0.82 |
| `crag_hallucination_rate` | 0.135 | 0.287 |
| `crag_accuracy` | 0.335 | 0.223 |

### Question-based features (9 columns)

Computed from the `question` column.

| Column | Method |
|--------|--------|
| `question_length_words` | Word count |
| `question_length_chars` | Character count |
| `question_complexity_score` | Flesch-Kincaid grade (`textstat`) |
| `has_few_shot_examples` | Regex: `Example 1:`, etc. |
| `prompt_contains_system_instructions` | Regex: `system:` or `Instructions:` |
| `question_category` | Rule-based: Reasoning / Fact Retrieval / Creative / Coding |
| `is_ambiguous` | True if prompt has fewer than 5 words |
| `contains_negation` | Keyword match (not, never, no, can't, …) |
| `context_token_count` | `tiktoken` (GPT-4) or word × 1.25 estimate (Llama) |

## 4. Model training

Predict `error` vs `no_error` with XGBoost on engineered features (WOE encoding for categoricals, median imputation for numerics).

```bash
python scripts/train_xgboost_woe.py
```

Or use the notebook: `notebooks/train_xgboost_woe.ipynb`.

### Pipeline

1. **Target:** `error` → 1, `no_error` → 0
2. **Split:** stratified 70 / 15 / 15 → 630 train · 135 val · 135 test
3. **Features:** 21 numeric + 6 boolean + 5 WOE-encoded categoricals = **32 features** (raw `question` excluded)
4. **Model:** XGBoost (300 trees, max_depth=4, `scale_pos_weight` for class imbalance)
5. **Artifacts:** saved to `models/`

### Results (test set)

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Train | 0.900 | 0.973 | 0.886 | 0.928 | 0.972 |
| Val | 0.696 | 0.786 | 0.794 | 0.790 | 0.758 |
| Test | 0.748 | 0.854 | 0.784 | 0.817 | 0.740 |

Test confusion matrix (rows = true, cols = predicted):

| | Pred no_error | Pred error |
|--|---------------|------------|
| True no_error | 25 | 13 |
| True error | 21 | 76 |

Train performance is much higher than val/test, indicating overfitting — partly because many features are constant per `llm_model`. Small val/test sets (135 rows each) also produce high metric variance.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/analyze_realmistake_full.ipynb` | EDA on base CSV: labels, models, tasks, prompt lengths |
| `notebooks/view_realmistake_enriched.ipynb` | Full view of all 35 enriched columns |
| `notebooks/train_xgboost_woe.ipynb` | Split → WOE → XGBoost → inference → scores |

Launch Jupyter:

```bash
jupyter notebook notebooks/
```

## End-to-end workflow

```bash
source venv/bin/activate

python scripts/download_realmistake.py
python scripts/build_full_dataset.py
python scripts/enrich_realmistake_full.py
python scripts/enrich_question_features.py
python scripts/train_xgboost_woe.py
```

## References

- Kamoi et al., [Evaluating LLMs at Detecting Errors in LLM Responses](https://arxiv.org/abs/2404.03602) (COLM 2024)
- Dataset: [ryokamoi/realmistake](https://huggingface.co/datasets/ryokamoi/realmistake)
- Code: [psunlpgroup/ReaLMistake](https://github.com/psunlpgroup/ReaLMistake)
