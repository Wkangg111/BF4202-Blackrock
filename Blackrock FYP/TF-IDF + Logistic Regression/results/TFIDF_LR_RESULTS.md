# TF-IDF + Logistic Regression — Escalation Model Results

Built using Wei Kang's exact prepared chronological splits (`prepared_data/splits/escalation/{train,validation,test}.csv`), so these numbers are directly comparable to his sentence-embedding + Logistic Regression results in the Sprint 2 summary / `MODEL_EVALUATION_REPORT.md`.

Methodology mirrored his approach:
- TF-IDF vectorizer fit on **training text only** (unigrams + bigrams, sublinear TF)
- `LogisticRegression(class_weight="balanced")`, C selected from `[0.01, 0.1, 1, 10, 100]` by validation average precision
- Alert threshold selected on validation set: minimize false positives subject to recall ≥ 90%
- Evaluated once on the untouched test set
- Second experiment adds one-hot encoded `category`/`subcategory`/`priority` (fit on training rows only, per `feature_policy.json`)

## Test set results

| Model | Precision | Recall | F1 | ROC AUC | False Positives | Missed Escalations |
|---|---:|---:|---:|---:|---:|---:|
| **TF-IDF + LR (text only)** | 85.93% | 88.60% | 87.24% | 0.6478 | 56 | 44 |
| Wei Kang — embedding + LR (text only) | 86.01% | 86.01% | 86.01% | 0.6508 | 54 | 54 |
| **TF-IDF + LR (text + structured)** | 87.79% | 87.56% | 87.68% | 0.7296 | 47 | 48 |
| Wei Kang — embedding + LR (text + structured) | 86.70% | 87.82% | 87.26% | 0.7212 | 52 | 47 |

## Takeaways

- **Text-only:** TF-IDF+LR gets noticeably better recall (88.60% vs 86.01%, i.e. it catches 10 more escalations) at the cost of 2 more false positives and essentially the same precision/F1. ROC AUC is basically tied.
- **Text + structured fields:** TF-IDF+LR edges out on precision, F1, and ROC AUC, and has 5 fewer false positives, while recall is a hair lower (87.56% vs 87.82%, 1 more missed escalation).
- Both approaches land in the same performance band — neither TF-IDF nor sentence embeddings is a clear winner here; the choice comes down to secondary factors: TF-IDF is simpler/faster to run and fully interpretable (you can inspect which words drove a score), while sentence embeddings generalize better to paraphrased wording the vocabulary hasn't seen.
- Adding structured fields (category/subcategory/priority) helps both approaches — it's the bigger lever here, not the text representation choice.
- This gives the team a legitimate head-to-head to put in the Sprint summary / slides: "we tried both text representations Wei Kang suggested, TF-IDF performs comparably to sentence embeddings."

Raw numbers: `tfidf_lr_summary.json`, `comparison_vs_wk.csv`.
