# Approach 4 (v2): sentence embeddings + linear SVM — ticket category & subcategory

Suggests the **category (17)** and **subcategory (59)** of a support ticket from the text the client writes
(short description + description), so it can be routed without manual triage. The text is encoded with the
frozen [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
encoder (384-d, normalised) and classified with a linear SVM (`sklearn.svm.LinearSVC`). A TF-IDF + SVM model is
always run alongside as the lexical baseline.

> **Data:** synthetic mock data (`source_data/`). Results are a prototype evaluation, not evidence of live
> routing accuracy. Keep the repo private if the data should not be shared.

## Folder layout

| Path | What it is |
|---|---|
| `source_data/` | `Case_details_synthetic_v5.xlsx` (used), `Escalation_data_synthetic_v2.xlsx`, `Interface_Catalog.xls` (not used, see below) |
| `text_cleaning.py` | Cleans and masks ticket text. Shared by data prep, the notebook and `predict.py` |
| `prepare_data.py` | Builds labels, removes label leakage, writes the chronological train/validation/test split |
| `data/` | `train.csv`, `validation.csv`, `test.csv`, `unlabelled.csv`, `label_hierarchy.csv`, `preparation_report.json` |
| `Approach_4_Category_SVM.ipynb` | The experiment: encoding, tuning, calibration, test, unseen-wording test, predictions |
| `model_utils.py` | Temperature-scaled probabilities for the SVM (needed to load the saved model) |
| `predict.py` | Suggest category/subcategory for new tickets with the saved model |
| `results/` | Metrics, predictions, charts, `category_model_svm.joblib`, `run_manifest.json` |
| `encoder/` | Local copy of MiniLM, created on first run (git-ignored) |

## How to run

Needs Python 3.10+ and internet access the first time (to download the encoder, about 90 MB).

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python prepare_data.py                                   # optional: data/ is already built
jupyter notebook Approach_4_Category_SVM.ipynb           # Kernel -> Restart & Run All, from this folder
```

The encoding cell must print **`MiniLM: AVAILABLE`**. If it prints `MiniLM: SKIPPED`, the reason is on the same
line (package not installed in the notebook's kernel, or huggingface.co blocked). In that case only the TF-IDF
baseline runs — there is **no stand-in**, and `results/run_manifest.json` records `ENCODER_BACKEND`. If your
network blocks Hugging Face, download the model elsewhere and put it in a folder named `encoder/`.

Predict new tickets:
```bash
python predict.py --short "Our positions don't line up" --description "Quantity difference on Instrument IS-12345 ..."
python predict.py --csv new_tickets.csv     # columns SHORT_DESCRIPTION, CASE_DESCRIPTION
```

## Method

1. **Target.** Subcategory (59 classes). Every subcategory has exactly one parent category (checked in
   `prepare_data.py`), so the category is read off the hierarchy and never contradicts the subcategory.
   A flat category model is also reported for comparison.
2. **Input.** Cleaned short description + description only. Excluded: assignment group (it is the routing
   target), priority, state, escalation and incident fields, client code, timestamps.
3. **Leakage removed from the text.** 2,425 service-form tickets contain `Request type: <subcategory>` and
   `Category: <category>` — the label itself — and these lines are stripped. Also removed: the support note naming
   a linked incident (added after triage), the quoted earlier message, greetings and sign-offs. IDs, amounts and
   dates are masked; the kind of ID (portfolio, fund, instrument…) is kept as a word.
4. **Split.** Chronological 70 / 13 / 17 % (Jan–early May / May / Jun 2024).
5. **Tuning.** `C` ∈ {0.01 … 30} and `class_weight` ∈ {None, balanced}, chosen by validation macro-F1. Test is
   never used for selection. Identical (text, label) rows are merged with a count weight (same model, ~20× faster).
6. **Confidence and auto-routing.** SVM margins → probabilities with temperature scaling, fitted on
   held-out-template cross-validation. A ticket is auto-routed when the top probability clears the threshold that
   gives ≥ 98 % accuracy on those held-out predictions; otherwise it goes to a person with the top-3 suggestions.
7. **Unseen-wording stress test.** The text is very templated: 200 short-description templates and ~1,000 unique
   masked texts across 28,750 tickets, and every validation/test template already appears in training. Grouped
   5-fold CV on train + validation holds out whole templates: (a) short description only, grouped by its template;
   (b) description body only, grouped by its template. **This decides which encoder is deployed.**

## Results (from the run in this folder — TF-IDF baseline only)

> The run included here was made where Hugging Face was blocked, so **MiniLM was skipped** (see
> `run_manifest.json`). The numbers below are the TF-IDF + SVM baseline. Re-run the notebook with MiniLM available
> to get the sentence-embedding rows. The comparison that matters is the unseen-wording test.

| Setting | Model | Subcategory acc. | Subcategory macro-F1 | Category acc. |
|---|---|---|---|---|
| Chronological test (Jun 2024, 4,888 tickets) | template lookup | 0.987 | 0.991 | 0.988 |
| | TF-IDF + SVM | 1.000 | 1.000 | 1.000 |
| Unseen **short description** (199 templates) | template lookup | 0.000 | 0.000 | 0.129 |
| | TF-IDF + SVM | 0.416 | 0.368 | 0.610 |
| Unseen **description body** (610 templates) | template lookup | 0.001 | 0.000 | 0.092 |
| | TF-IDF + SVM | 0.821 | 0.686 | 0.906 |

How to read this:
* The **chronological test is saturated**. Looking up the short description in a table already scores 98.7 %, so
  100 % says the model memorised the templates, not that it understands requests. Don't quote it on its own.
* **New wording is where the models differ.** TF-IDF falls to 0.37 / 0.69 macro-F1, because it relies on matching
  words it has seen. A sentence encoder should do better here, since it maps paraphrases to nearby vectors.
  That is the hypothesis the MiniLM run tests.
* Most errors on new wording are between semantically close subcategories, e.g. *Cash balance query* vs
  *Client account query*, *Benchmark data delay* vs *Benchmark addition*, *Holdings mismatch* vs
  *Position break* / *NAV difference* (`results/unseen_wording_top_confusions.csv`).
* **Auto-routing.** On full text with a new short description, held-out accuracy is 98.3 % before any threshold,
  so the 98 % target is met with threshold 0 and all 600 unlabelled tickets are auto-routed
  (`results/unlabelled_predictions.csv`). On real, less templated data the same procedure will set a higher
  threshold. Raise `TARGET_AUTO_ACCURACY` to send more tickets to a human.

## Inputs not used, and why

* **`Escalation_data_synthetic_v2.xlsx`** — escalations are raised after a ticket is categorised, so they cannot
  inform the category at creation time.
* **`Interface_Catalog.xls`** — a catalogue of Aladdin interface files/tags. No ticket mentions an interface name
  or spec number, so it adds no signal on this data. With real tickets that cite interfaces (e.g. "file 14
  Positions not received"), it could add keyword features or map interfaces to *Data Delivery* subcategories.

## Limitations

Synthetic, heavily templated data; labels assumed correct; MiniLM-L6 is English-only, so the 558 Japanese tickets
rely on the English short description (`paraphrase-multilingual-MiniLM-L12-v2` is a drop-in alternative);
probabilities are calibrated on held-out templates from the same synthetic generator.
