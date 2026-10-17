# TF-IDF + Logistic Regression (escalation model)

This folder holds an alternative escalation-risk model, built as a direct comparison to the
`Pretrained sentence-embedding + logistic regression` model in this repo. Same classifier
(Logistic Regression), same prepared data, same evaluation methodology — the only thing that
differs is how ticket text is turned into numbers: TF-IDF (word-frequency based) here, versus
sentence embeddings (meaning-based) in the other folder.

## Data

Uses the shared `../prepared_data/splits/escalation/{train,validation,test}.csv` produced by
`../prepare_data.py` — no separate data preparation step. This keeps the comparison fair: both
models train and evaluate on identical rows and identical chronological split boundaries.

Structured-field experiment additionally uses
`../Pretrained sentence-embedding + logistic regression/data/assumed_initial_fields.csv`
(category / subcategory / priority, joined on `ticket_id`).

## Method

- `TfidfVectorizer` fit on **training text only** (unigrams + bigrams, sublinear TF) — vocabulary
  is never exposed to validation/test text, consistent with `feature_policy.json`'s
  `fit_after_split` rule.
- `LogisticRegression(class_weight="balanced")`, with `C` selected from `[0.01, 0.1, 1, 10, 100]`
  by validation average precision — same selection rule documented in the embedding model's
  `MODEL_EVALUATION_REPORT.md`.
- Alert threshold selected on the validation set only: minimize false positives subject to
  recall >= 90%.
- Evaluated once, on the untouched test set.
- Repeated with `category` / `subcategory` / `priority` one-hot encoded and concatenated onto the
  TF-IDF matrix (encoder fit on training rows only).

## How to run

```bash
pip install -r requirements.txt
python tfidf_lr_escalation.py
```

Expects the repo layout as checked in (relative paths up to `../prepared_data/` and
`../Pretrained sentence-embedding + logistic regression/data/`). Outputs land in `results/`:
`tfidf_lr_summary.json` (full metrics) and `comparison_vs_wk.csv` (side-by-side vs. the
embedding model's reported numbers).

## Results (test set)

| Model | Precision | Recall | F1 | ROC AUC | False Positives | Missed Escalations |
|---|---:|---:|---:|---:|---:|---:|
| **TF-IDF + LR (text only)** | 85.93% | 88.60% | 87.24% | 0.6478 | 56 | 44 |
| Embedding + LR (text only) | 86.01% | 86.01% | 86.01% | 0.6508 | 54 | 54 |
| **TF-IDF + LR (text + structured)** | 87.79% | 87.56% | 87.68% | 0.7296 | 47 | 48 |
| Embedding + LR (text + structured) | 86.70% | 87.82% | 87.26% | 0.7212 | 52 | 47 |

Full discussion of these results is in `results/TFIDF_LR_RESULTS.md`.

## Takeaway

TF-IDF and sentence embeddings perform comparably on this dataset — neither is a clear winner.
TF-IDF text-only gets better recall; adding structured fields, TF-IDF edges ahead on precision/F1/
false positives. TF-IDF is simpler, faster, and fully interpretable (inspectable word weights),
which is useful for the project's "insufficient information" flagging requirement. Sentence
embeddings would be expected to pull further ahead at larger scale with more varied/global
phrasing — see project notes for the longer discussion of that tradeoff.
