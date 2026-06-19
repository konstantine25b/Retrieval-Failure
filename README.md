# Retrieval Failure

Analysis and modeling pipeline for LLM error detection across multiple benchmarks: [ReaLMistake](https://arxiv.org/abs/2404.03602) (900 expert-annotated examples), [Mis-prompt](https://arxiv.org/abs/2506.00064) (29,392 proactive error-handling prompts), and [MMLU-Pro](https://arxiv.org/abs/2406.01574) (563,787 model–question outcomes from 47 LLMs).

## Project structure

```
.
├── data/
│   ├── realmistake/                  # Source JSONL (3 tasks × 2 models)
│   ├── misprompt/                    # Mis-prompt JSON (downloaded, gitignored)
│   ├── mmlu_pro/                     # MMLU-Pro eval JSON (downloaded)
│   ├── realmistake_full.csv          # Base tabular export (900 rows)
│   ├── misprompt_full.csv            # Mis-prompt export (29,392 rows)
│   ├── mmlu_pro_full.csv             # MMLU-Pro export (563,787 rows)
│   ├── mmlu_pro_full_enriched.csv    # MMLU-Pro enriched (563,787 × 35 cols)
│   ├── combined_full.csv             # ReaLMistake + Mis-prompt + MMLU-Pro (594,079 rows)
│   ├── realmistake_full_enriched.csv # Full feature set (900 rows × 35 cols)
│   ├── misprompt_full_enriched.csv   # Mis-prompt enriched (29,392 × 41 cols)
│   └── combined_full_enriched.csv    # Combined enriched (30,292 × 42 cols; regenerate after adding MMLU-Pro)
├── scripts/
│   ├── download_realmistake.py       # Download dataset from Hugging Face or zip
│   ├── download_misprompt.py         # Download Mis-prompt from GitHub
│   ├── download_mmlu_pro.py          # Download MMLU-Pro eval results from GitHub
│   ├── build_full_dataset.py         # JSONL → realmistake_full.csv
│   ├── build_misprompt_dataset.py    # Mis-prompt JSON → misprompt_full.csv
│   ├── build_mmlu_pro_dataset.py     # MMLU-Pro eval JSON → mmlu_pro_full.csv
│   ├── enrich_mmlu_pro_full.py       # Enrich MMLU-Pro CSV (model + question features)
│   ├── model_features.py             # Per-model architecture/hyperparam lookup (47 MMLU models)
│   ├── build_combined_dataset.py     # Merge ReaLMistake + Mis-prompt + MMLU-Pro
│   ├── enrich_realmistake_full.py      # Add model-level features
│   ├── enrich_misprompt_full.py      # Enrich Mis-prompt CSV
│   ├── enrich_combined_full.py       # Enrich combined CSV
│   ├── enrich_question_features.py   # Add question-based features
│   ├── train_xgboost_woe.py          # WOE + XGBoost training & evaluation
│   ├── train_xgboost_woe_iv_rfe.py   # WOE + IV + RFE + XGBoost
│   ├── train_xgboost_combined_woe_iv_rfe.py  # Combined data WOE + IV + RFE + XGBoost
│   ├── train_xgboost_combined_woe_iv_rfe_complex.py  # Combined data, deeper XGBoost
│   └── train_random_forest_woe.py    # WOE + Random Forest training & evaluation
├── notebooks/
│   ├── analyze_realmistake_full.ipynb
│   ├── view_realmistake_enriched.ipynb
│   ├── train_xgboost_woe.ipynb
│   ├── train_xgboost_woe_iv_rfe.ipynb
│   ├── train_xgboost_combined_woe_iv_rfe.ipynb
│   ├── train_xgboost_combined_woe_iv_rfe_complex.ipynb
│   └── train_random_forest_woe.ipynb
├── models/
│   ├── xgboost_woe.json              # Trained XGBoost classifier
│   ├── xgboost_woe_iv_rfe.json       # WOE + IV + RFE XGBoost classifier
│   ├── xgboost_combined_woe_iv_rfe.json  # Combined data WOE + IV + RFE XGBoost
│   ├── xgboost_combined_woe_iv_rfe_complex.json  # Combined data, deeper XGBoost
│   ├── random_forest_woe.joblib      # Trained Random Forest classifier
│   ├── preprocessing.json            # XGBoost WOE maps & imputation values
│   ├── xgboost_woe_iv_rfe_preprocessing.json
│   ├── xgboost_combined_woe_iv_rfe_preprocessing.json
│   ├── xgboost_combined_woe_iv_rfe_complex_preprocessing.json
│   ├── random_forest_preprocessing.json
│   ├── metrics.json                  # XGBoost train / val / test scores
│   ├── xgboost_woe_iv_rfe_metrics.json
│   ├── xgboost_combined_woe_iv_rfe_metrics.json
│   ├── xgboost_combined_woe_iv_rfe_complex_metrics.json
│   └── random_forest_metrics.json    # Random Forest train / val / test scores
└── requirements.txt
```

## Setup

```bash
cd "Retrival Failure"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Sample data

Examples below are truncated for readability. Full rows live in `data/realmistake/`, `data/realmistake_full.csv`, and `data/realmistake_full_enriched.csv`.

### Source JSONL (`data/realmistake/example.jsonl`)

```json
{
  "input": "Generate a math word problem that satisfies the following requirements. First, provide the generated question. Second, generate a step-by-step solution...\n\nSpecific Requirements:\n* The problem statement should include the phrase \"Marla completes 32 laps around the track per hour\".\n* The solution involves division and rounding off to the nearest whole number.",
  "llm_response": "Question:\nMarla completes 32 laps around the track per hour. If the track is 400 meters long, how many kilometers does Marla run in 2.5 hours?\n\nSolution:\nStep 1: ...\nSo, Marla runs 32 kilometers in 2.5 hours.",
  "error_label": "error",
  "human_explanation": "The question does not involve relative speed and rounding off to the nearest whole number, which are in the requirements. In addition, the solution does not include numbers with two decimal places.",
  "error_categories": ["Instruction-Following"],
  "metadata": {
    "id": "math_problem_generation_66_gpt-4-0613",
    "task_name": "math_problem_generation",
    "task_source": "aqua",
    "llm_response_model": "gpt-4-0613",
    "dataset": "realmistake",
    "difficulty": "difficult"
  }
}
```

### Base CSV (`data/realmistake_full.csv`)

| question | llm_model | error |
|----------|-----------|-------|
| Generate a math word problem that satisfies the following requirements. First, provide the generated question. Second, generate a step-by-step solution for the generated question… *(1,758 chars)* | `meta-llama/Llama-2-70b-chat-hf` | `error` |
| We provide a pair of a claim and evidence. The claim is a sentence in a Wikipedia article… *(3,500+ chars)* | `gpt-4-0613` | `error` |
| Answer the following question. Assume you are on Jan 18, 2018 and questions that require knowledge after this date should be classified as unanswerable… *(744 chars)* | `gpt-4-0613` | `no_error` |

### Enriched CSV (`data/realmistake_full_enriched.csv`)

Same rows as above, plus 32 engineered features. Example (math word problem, Llama 2 70B):

| Column | Value |
|--------|-------|
| `question` | Generate a math word problem that satisfies the following requirements… *(truncated)* |
| `llm_model` | `meta-llama/Llama-2-70b-chat-hf` |
| `error` | `error` |
| `model_name` | `llama-2-70b-chat-hf` |
| `context_window_tokens` | 4096 |
| `max_output_tokens` | 2048 |
| `attention_type` | GQA |
| `temperature` | 0.6 |
| `galileo_qa_no_rag` | 0.65 |
| `crag_hallucination_rate` | 0.287 |
| `question_length_words` | 274 |
| `question_length_chars` | 1758 |
| `question_complexity_score` | 10.51 |
| `question_category` | Reasoning |
| `contains_negation` | True |
| `context_token_count` | 342 |
| `has_few_shot_examples` | False |

Example (fact verification, Llama 2 70B):

| Column | Value |
|--------|-------|
| `question` | We provide a pair of a claim and evidence. The claim is a sentence in a Wikipedia article… *(truncated)* |
| `llm_model` | `meta-llama/Llama-2-70b-chat-hf` |
| `error` | `error` |
| `question_category` | Fact Retrieval |
| `question_length_words` | 549 |
| `context_token_count` | 686 |
| `galileo_qa_no_rag` | 0.65 |
| `crag_accuracy` | 0.223 |

### Dataset summary

| | Count |
|--|-------|
| Total rows | 900 |
| `error` | 649 (72.1%) |
| `no_error` | 251 (27.9%) |
| GPT-4-0613 | 420 |
| Llama 2 70B | 480 |
| Reasoning prompts | 585 |
| Fact Retrieval prompts | 241 |

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

## 2b. Mis-prompt dataset (ACL 2025)

Proactive error-handling benchmark: flawed user prompts with expert labels and gold responses.

```bash
python scripts/download_misprompt.py
python scripts/build_misprompt_dataset.py
python scripts/enrich_misprompt_full.py
```

Source: [Jiayi-Zeng/mis-prompt](https://github.com/Jiayi-Zeng/mis-prompt)

Creates `data/misprompt_full.csv` (29,392 rows from train/dev/eval splits):

| Column | Description |
|--------|-------------|
| `question` | User prompt (`prompt` in source JSON) |
| `llm_model` | `gpt-4o` (Mis-prompt data generated with GPT-4o) |
| `error` | `error` (14,696) or `no_error` (14,696) — source uses `correct` → `no_error` |
| `split` | `train` / `dev` / `eval` |
| `primary_category` | Language / Incomplete / Factual / Logical Errors |
| `secondary_category` | Fine-grained error type (14 categories) |
| `explanation` | Why the prompt is wrong (error rows only) |
| `gold_answer` | Ideal proactive error-handling response |

## 2c. MMLU-Pro dataset (NeurIPS 2024)

Multiple-choice benchmark with 10 options per question. We use the official cached model predictions (not just the question bank) so each row records whether a specific LLM answered correctly.

```bash
python scripts/download_mmlu_pro.py
python scripts/build_mmlu_pro_dataset.py
```

Sources:

- Questions: [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
- Model predictions: [TIGER-AI-Lab/MMLU-Pro eval_results](https://github.com/TIGER-AI-Lab/MMLU-Pro/tree/main/eval_results)

Creates `data/mmlu_pro_full.csv` (563,787 rows from 47 models × ~12,032 questions):

| Column | Description |
|--------|-------------|
| `question` | Question stem + options A–J (newline-separated) |
| `llm_model` | Model that answered (e.g. `gpt-4o-2024-08-06`, `Llama-2-7b-hf`) |
| `error` | `no_error` if model answer matches gold, else `error` (includes missing/unparseable answers) |
| `id` | MMLU-Pro `question_id` |
| `category` | Subject (e.g. `math`, `physics`, `law`) |
| `gold_answer` | Correct option letter (A–J) |
| `model_answer` | Model's chosen letter, or empty if no answer |

### Example — correct row (`no_error`)

```
question:
What will be the number of lamps, each having 300 lumens, required to obtain an average
illuminance of 50 lux on a 4m * 3m rectangular room?

A. 6
B. 1
C. 2
...

llm_model:    DeepSeek-Coder-V2
error:         no_error
id:            11285
category:      engineering
gold_answer:   C
model_answer:  C
```

### Example — wrong row (`error`)

```
question:
In force-current analogy, electrical analogous quantity for displacement (x) is

A. inductance.
B. resistance.
...
I. flux.
J. current.

llm_model:    DeepSeek-Coder-V2
error:         error
id:            11289
category:      engineering
gold_answer:   I
model_answer:  G
```

Each question appears once per model. The same `id` can therefore appear up to 47 times with different `llm_model` / `error` values.

### MMLU-Pro models (47)

Exact `llm_model` values as stored in `data/mmlu_pro_full.csv`:

1. `DeepSeek-Coder-V2`
2. `Llama-2-13b-hf`
3. `Llama-2-70b-hf`
4. `Llama-2-7b-hf`
5. `Meta-Llama-3-70B`
6. `Meta-Llama-3-70B-Instruct`
7. `Meta-Llama-3-8B`
8. `Meta-Llama-3-8B-Instruct`
9. `Meta-Llama-3_1-70B`
10. `Meta-Llama-3_1-70B-Instruct`
11. `Meta-Llama-3_1-8B`
12. `Meta-Llama-3_1-8B-Instruct`
13. `Mistral-7B-Instruct-v0.1`
14. `Mistral-7B-Instruct-v0.2`
15. `Mistral-7B-v0.1`
16. `Mistral-7B-v0.2-hf`
17. `Mixtral-8x7B-Instruct-v0.1`
18. `Mixtral-8x7B-v0.1`
19. `Phi-3-mini-4k-instruct`
20. `Qwen1.5-110B`
21. `Qwen1.5-14B-Chat`
22. `Qwen1.5-72B-Chat`
23. `Qwen1.5-7B-Chat`
24. `Yi-34B`
25. `Yi-6B`
26. `Yi-6b-Chat`
27. `arx_0314`
28. `arx_3`
29. `c4ai-command-r-v01`
30. `claude-3-5-haiku-20241022`
31. `claude-3-5-sonnet-20241022`
32. `claude-3.5-sonnet`
33. `deepseek`
34. `deepseek-chat-v2_5`
35. `flash_0shots_00_35_03`
36. `gemini-1.5-flash-002`
37. `gemini-1.5-pro-002`
38. `gemma-7b`
39. `gpt-4o-2024-08-06`
40. `gpt-4o-mini`
41. `gpt4o(2024-05-13)`
42. `iask_pro`
43. `jamba-1.5-large`
44. `mathstral-7B`
45. `opus_2shots_00_37_14`
46. `sonnet-3.5_0shots_09_34_29`
47. `sonnet_0shots_12_01_18`

### MMLU-Pro summary

| | Count |
|--|-------|
| Total rows | 563,787 |
| Models | 47 |
| `no_error` (correct) | 280,943 (49.8%) |
| `error` (wrong / no answer) | 282,844 (50.2%) |

### Enrich MMLU-Pro for training

Build a **standalone** enriched dataset (not merged with ReaLMistake/Mis-prompt):

```bash
python scripts/enrich_mmlu_pro_full.py
```

Output: `data/mmlu_pro_full_enriched.csv` — **563,787 rows × 35 columns**.

**Base columns (target + identifiers):**

| Column | Description |
|--------|-------------|
| `question` | Multiple-choice question + options A–J |
| `llm_model` | One of 47 models (see list above) |
| `error` | `no_error` (correct) or `error` (wrong / no answer) |

**Model-level features (23 columns)** — joined by `llm_model`, unique per model from `scripts/model_features.py`:

| Column | Examples across models |
|--------|------------------------|
| `context_window_tokens` | 4096 (Llama-2-7b) · 8192 (Llama-3-8B) · 131072 (DeepSeek-Coder-V2) · 128000 (GPT-4o) |
| `attention_type` | MHA · GQA · MoE · hybrid_Mamba |
| `tokenizer_type` | sentencepiece_BPE · cl100k_BPE · bytelevel_BPE |
| `is_open_source` | True (Llama, Qwen, Mistral) · False (GPT-4o, Claude, Gemini) |
| `temperature`, `top_p`, `top_k` | Model-specific API defaults |
| `galileo_qa_no_rag`, `crag_hallucination_rate` | External benchmark proxies (tiered by model capability) |

**Question-level features (9 columns)** — computed per row from `question`:

| Column | Method |
|--------|--------|
| `question_length_words` | Word count |
| `question_length_chars` | Character count |
| `question_complexity_score` | Flesch-Kincaid grade (`textstat`) |
| `question_category` | MMLU subject (math, physics, law, …) |
| `context_token_count` | `tiktoken` for API models, word × 1.25 estimate for open-weight models |
| `contains_negation`, `has_few_shot_examples`, … | Regex flags |

Each of the 47 models has its own feature row (context window, attention type, tokenizer, hyperparams, benchmark scores). Use `question`, `llm_model`, and the 32 feature columns to predict `error`.

Combine with ReaLMistake and Mis-prompt (optional):

```bash
python scripts/build_combined_dataset.py
python scripts/enrich_combined_full.py
```

Output: `data/combined_full.csv` (594,079 rows). Regenerate `combined_full_enriched.csv` after merging — the existing enriched file predates MMLU-Pro.

| Source | Rows | error | no_error |
|--------|------|-------|----------|
| ReaLMistake | 900 | 649 | 251 |
| Mis-prompt | 29,392 | 14,696 | 14,696 |
| MMLU-Pro | 563,787 | 282,844 | 280,943 |
| Combined | 594,079 | 298,189 | 295,890 |

In `combined_full.csv`, MMLU-Pro rows have `source = "mmlu_pro"`, `primary_category` set to the subject, and `model_answer` dropped.

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

Predict `error` vs `no_error` on engineered features (WOE encoding for categoricals, median imputation for numerics).

### Pipeline (shared)

1. **Target:** `error` → 1, `no_error` → 0
2. **Split:** stratified 70 / 15 / 15 → 630 train · 135 val · 135 test
3. **Features:** 21 numeric + 6 boolean + 5 WOE-encoded categoricals = **32 features** (raw `question` excluded)
4. **Artifacts:** saved to `models/`

### XGBoost

```bash
python scripts/train_xgboost_woe.py
```

Or use the notebook: `notebooks/train_xgboost_woe.ipynb`.

- **Model:** XGBoost (300 trees, max_depth=4, `scale_pos_weight` for class imbalance)

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

### XGBoost + WOE + IV + RFE

```bash
python scripts/train_xgboost_woe_iv_rfe.py
```

Or use the notebook: `notebooks/train_xgboost_woe_iv_rfe.ipynb`.

- **Feature selection:** IV threshold 0.02 → 10 features, then RFE → 10 features
- **Model:** XGBoost (300 trees, max_depth=4, `scale_pos_weight`)

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Train | 0.903 | 0.974 | 0.890 | 0.930 | 0.972 |
| Val | 0.704 | 0.794 | 0.794 | 0.794 | 0.756 |
| Test | 0.756 | 0.848 | 0.804 | 0.825 | 0.739 |

Selected features: `question_length_words`, `question_length_chars`, `question_complexity_score`, `context_token_count`, `is_open_source`, `multilingual_support`, `model_name_woe`, `positional_encoding_type_woe`, `attention_type_woe`, `tokenizer_type_woe`

Test confusion matrix (rows = true, cols = predicted):

| | Pred no_error | Pred error |
|--|---------------|------------|
| True no_error | 24 | 14 |
| True error | 19 | 78 |

### XGBoost + WOE + IV + RFE (Combined data)

```bash
python scripts/train_xgboost_combined_woe_iv_rfe.py
```

Or use the notebook: `notebooks/train_xgboost_combined_woe_iv_rfe.ipynb`.

- **Data:** `combined_full_enriched.csv` (30,292 rows — ReaLMistake + Mis-prompt)
- **Split:** stratified 70 / 15 / 15 → 21,204 train · 4,544 val · 4,544 test
- **Feature selection:** IV threshold 0.02 → 11 features, then RFE → 11 features
- **Model:** XGBoost (300 trees, max_depth=4, `scale_pos_weight`)

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Train | 0.767 | 0.793 | 0.731 | 0.761 | 0.849 |
| Val | 0.749 | 0.766 | 0.726 | 0.745 | 0.831 |
| Test | 0.759 | 0.781 | 0.727 | 0.753 | 0.829 |

Selected features: `question_length_words`, `question_length_chars`, `question_complexity_score`, `context_token_count`, `is_open_source`, `multilingual_support`, `contains_negation`, `model_name_woe`, `positional_encoding_type_woe`, `attention_type_woe`, `tokenizer_type_woe`

Test confusion matrix (rows = true, cols = predicted):

| | Pred no_error | Pred error |
|--|---------------|------------|
| True no_error | 1773 | 469 |
| True error | 628 | 1674 |

### XGBoost + WOE + IV + RFE (Combined, complex model)

```bash
python scripts/train_xgboost_combined_woe_iv_rfe_complex.py
```

Or use the notebook: `notebooks/train_xgboost_combined_woe_iv_rfe_complex.ipynb`.

Same data, split, and feature selection as the baseline combined model. XGBoost is deeper and regularized:

- **Model:** 1000 trees (early stopping at 655), max_depth=7, lr=0.02, subsample/colsample=0.8, min_child_weight=3, gamma=0.1, L1/L2 reg

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Train | 0.790 | 0.810 | 0.763 | 0.786 | 0.874 |
| Val | 0.753 | 0.767 | 0.737 | 0.752 | 0.834 |
| Test | 0.756 | 0.775 | 0.732 | 0.752 | **0.832** |

Test confusion matrix (rows = true, cols = predicted):

| | Pred no_error | Pred error |
|--|---------------|------------|
| True no_error | 1752 | 490 |
| True error | 618 | 1684 |

Compared to the baseline combined model, the complex variant improves val F1 (+0.7 pp) and test ROC-AUC (+0.3 pp) with a smaller train–test gap, at the cost of slightly lower test accuracy (−0.3 pp).

### Random Forest

```bash
python scripts/train_random_forest_woe.py
```

Or use the notebook: `notebooks/train_random_forest_woe.ipynb`.

- **Model:** Random Forest (300 trees, max_depth=4, `class_weight='balanced'`)

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Train | 0.721 | 0.929 | 0.664 | 0.774 | 0.875 |
| Val | 0.622 | 0.788 | 0.649 | 0.712 | 0.727 |
| Test | 0.652 | 0.857 | 0.619 | 0.719 | 0.688 |

Test confusion matrix (rows = true, cols = predicted):

| | Pred no_error | Pred error |
|--|---------------|------------|
| True no_error | 28 | 10 |
| True error | 37 | 60 |

### Model comparison (test set)

| Model | Data | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|------|----------|-----------|--------|-----|---------|
| XGBoost + WOE + IV + RFE (Combined, complex) | 30,292 | 0.756 | 0.775 | 0.732 | 0.752 | **0.832** |
| XGBoost + WOE + IV + RFE (Combined) | 30,292 | **0.759** | **0.781** | 0.727 | **0.753** | 0.829 |
| XGBoost + WOE + IV + RFE | 900 | 0.756 | **0.848** | **0.804** | **0.825** | 0.739 |
| XGBoost | 900 | 0.748 | 0.854 | 0.784 | 0.817 | 0.740 |
| Random Forest | 900 | 0.652 | 0.857 | 0.619 | 0.719 | 0.688 |

Combined models generalize better (ROC-AUC ~0.83 vs 0.739) with a much larger test set (4,544 vs 135 rows). The complex XGBoost slightly improves ranking (ROC-AUC 0.832) and val performance; the baseline combined model still has marginally higher test accuracy/F1. ReaLMistake-only models achieve higher precision/recall on their small test split but show larger train–test gaps.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/analyze_realmistake_full.ipynb` | EDA on base CSV: labels, models, tasks, prompt lengths |
| `notebooks/view_realmistake_enriched.ipynb` | Full view of all 35 enriched columns |
| `notebooks/train_xgboost_woe.ipynb` | Split → WOE → XGBoost → inference → scores |
| `notebooks/train_xgboost_woe_iv_rfe.ipynb` | Split → WOE → IV → RFE → XGBoost → scores |
| `notebooks/train_xgboost_combined_woe_iv_rfe.ipynb` | Combined data → WOE → IV → RFE → XGBoost → scores |
| `notebooks/train_xgboost_combined_woe_iv_rfe_complex.ipynb` | Combined data → WOE → IV → RFE → deep XGBoost → scores |
| `notebooks/train_random_forest_woe.ipynb` | Split → WOE → Random Forest → inference → scores |

Launch Jupyter:

```bash
jupyter notebook notebooks/
```

## End-to-end workflow

```bash
source venv/bin/activate

python scripts/download_realmistake.py
python scripts/build_full_dataset.py
python scripts/download_misprompt.py
python scripts/build_misprompt_dataset.py
python scripts/download_mmlu_pro.py
python scripts/build_mmlu_pro_dataset.py
python scripts/enrich_mmlu_pro_full.py
python scripts/build_combined_dataset.py
python scripts/enrich_realmistake_full.py
python scripts/enrich_misprompt_full.py
python scripts/enrich_combined_full.py
python scripts/enrich_question_features.py
python scripts/train_xgboost_woe.py
python scripts/train_xgboost_woe_iv_rfe.py
python scripts/train_xgboost_combined_woe_iv_rfe.py
python scripts/train_xgboost_combined_woe_iv_rfe_complex.py
python scripts/train_random_forest_woe.py
```

## References

- Kamoi et al., [Evaluating LLMs at Detecting Errors in LLM Responses](https://arxiv.org/abs/2404.03602) (COLM 2024)
- Zeng et al., [Mis-prompt: Benchmarking Large Language Models for Proactive Error Handling](https://arxiv.org/abs/2506.00064) (ACL 2025)
- Wang et al., [MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark](https://arxiv.org/abs/2406.01574) (NeurIPS 2024)
- Dataset: [ryokamoi/realmistake](https://huggingface.co/datasets/ryokamoi/realmistake)
- Dataset: [Jiayi-Zeng/mis-prompt](https://github.com/Jiayi-Zeng/mis-prompt)
- Dataset: [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
- Code: [psunlpgroup/ReaLMistake](https://github.com/psunlpgroup/ReaLMistake)
- Code: [TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro)
