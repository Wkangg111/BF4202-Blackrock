# Approach 4: sentence embeddings + linear SVM (escalation risk)

Predicts whether a support ticket will be escalated, using only its cleaned text (and, in a second experiment,
category / subcategory / assumed initial priority). The text is encoded with the frozen
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
encoder (384-d, normalised) and classified with a linear SVM (`sklearn.svm.LinearSVC`).

> **Data:** synthetic mock data (`source_data/`). Results are a prototype evaluation, not evidence of live
> early-warning accuracy. The repo is private; keep it that way if the data should not be shared.

## Folder layout

| Path | What it is |
|---|---|
| `source_data/` | Raw synthetic Excel exports (cases, escalations, incidents) |
| `prepare_data.py` | Cleans text, builds the label and groups, writes the chronological train/validation/test split |
| `data/` | Prepared `train.csv`, `validation.csv`, `test.csv`, `assumed_initial_fields.csv`, `preparation_report.json` |
| `Approach_4_Escalation_SVM.ipynb` | The experiment: encoding, SVM training, evaluation, grouped cross-validation, results highlights |
| `results/` | Metrics CSVs, predictions, charts, saved models (`escalation_model_svm*.joblib`), `run_manifest.json` |
| `encoder/` | Local copy of the encoder, created on first run (git-ignored) |

## How to run (redo the model with sentence embeddings)

Needs Python 3.10+ and internet access the first time (to download the encoder, about 90 MB).

1. **Clone and enter the repo**
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```
2. **Install packages** (a virtual environment is optional but tidy)
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **(Optional) rebuild the prepared data** from `source_data/`. `data/` is already committed, so this only
   regenerates the same files.
   ```bash
   python prepare_data.py
   ```
4. **Open the notebook** from the repo root (it must be the working directory, otherwise the first cell raises an error)
   ```bash
   jupyter notebook Approach_4_Escalation_SVM.ipynb
   ```
5. **Run all cells** (Kernel -> Restart & Run All). The first run downloads `all-MiniLM-L6-v2` into `encoder/`;
   later runs reuse it. Encoding about 1,200 unique texts on CPU takes a few minutes.
6. **Confirm the real encoder was used.** The encoding cell must print
   `ENCODER BACKEND: sentence-transformers`. If it prints `STAND-IN: tfidf+svd`, the package or download failed
   (the line above shows why). The stand-in exists only so the pipeline can still run; its numbers are **not**
   sentence-embedding results. Common causes: a firewall/proxy blocking huggingface.co, or the notebook kernel
   not using the environment where `sentence-transformers` was installed. If your network blocks Hugging Face,
   download the model elsewhere and place it in a folder named `encoder/`; the notebook uses it when present.
7. **Read the results** in the "Results highlights" section at the end of the notebook. Tables are rebuilt from
   the run, and `results/` is overwritten. Also check `results/run_manifest.json` shows
   `"ENCODER_BACKEND": "sentence-transformers"`.

## What the notebook does

1. **Text-only SVM, F1 threshold.** C is chosen by validation average precision; the alert threshold by validation F1.
2. **Text-only SVM, recall >= 90% threshold.** Minimises validation false positives subject to 90% recall.
3. **Text + category + subcategory + assumed initial priority.** Same selection policy. Assumes the supplied fields are the initial values (unverified).
4. **Nested five-fold grouped cross-validation.** Tickets sharing text or an incident/problem group stay in the same fold.

Test data is never used to select C or thresholds. Scores are SVM margins (not probabilities), so the default
threshold is 0 and no Brier score is reported. Always-alert and always-negative benchmarks are included because
about 85% of tickets are escalated, which makes F1 alone misleading.

## Notes

- `results/embeddings.npz` and `encoder/` are git-ignored (regenerated on each run).
- Stored results in `results/` and in the committed notebook may come from the stand-in encoder unless you re-ran
  the notebook with the real one. Check `run_manifest.json` before quoting any numbers.
