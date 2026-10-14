# Final ticket data preparation

This package contains the three finalised source Excel files, a reproducible Python preparation script, generated CSV outputs, an audit report and regression tests. It does not train a model or modify the source workbooks.

## Run in VS Code

1. Extract the ZIP and open `ticket_data_preparation` as a folder in VS Code.
2. Select your Python 3.11 or 3.12 interpreter.
3. In the VS Code integrated terminal, install dependencies:

```sh
python -m pip install -r requirements.txt
```

4. Open `prepare_data.py` and click **Run Python File**, or run:

```sh
python prepare_data.py
```

No mandatory command-line dates are required. Default paths are relative to the script, so the Run button works from another working directory. Results are written to `prepared_data`.

5. Check the saved results:

```sh
python -m unittest -v test_preparation.py
```

The supplied-output tests check this exact final dataset. Adapt fixed row-count expectations when switching to a new dataset.

## Which files should train the model?

For a retrospective synthetic prototype, use `prepared_data/splits/escalation/train.csv`, `validation.csv` and `test.csv`.

- X: **model_text**.
- y: **escalation_label**.
- The remaining ID, time and group columns are metadata, not predictors.
- Fit the TF-IDF vectorizer and classifier on training rows only. Use validation for model/threshold choices; evaluate the untouched test set once.
- Every escalation counts equally. Never filter events by approval status.
- Multiple escalation records for one case produce one positive ticket, not repeated training rows.

Folders for category, subcategory, priority and team supply separate target columns. Do not combine tasks' labels as model inputs. Missing targets and unresolved team assignments are excluded from the corresponding task, not from every task.

## Why two sets of splits?

`splits/` is **retrospective**: tickets are ordered by creation date, but outcomes may become known after historical partition boundaries. This permits prototype training using the completed export. It is not evidence that the model could have been built at those historical dates.

`strict_splits/` additionally requires labels to be known by the original training/validation cutoff. These files reveal insufficient follow-up in this one-month ticket sample: validation is empty for some tasks and contains only positives for escalation. Do not use those sets for model selection as-is. A strict prospective evaluation needs a longer period, adequate follow-up and verified historical snapshots.

Both sets exclude ticket groups linked through incidents sharing a problem number when a group spans creation-date boundaries. The optional `validation_unseen_text.csv` and `test_unseen_text.csv` files under `splits/` exclude exact model texts appearing in earlier primary partitions. They are diagnostic subsets and may be small; they do not remove every paraphrased synthetic template.

## Snapshot and incident limitations

These final files have no initial-snapshot sheet or edit history. The script removes explicit support notes and outcome identifiers, then treats the remaining request text as a candidate receipt-time input. **This assumption cannot be verified from the export.** Default features are text-only. Review the original data source before making a real early-warning claim.

Current category, subcategory, priority and team values are reviewed targets, not automatically valid initial predictors. An explicit `--assume-initial-fields` flag adds category/subcategory/priority/peer group to the escalation splits for a documented experiment. Do not use this flag unless their receipt-time availability is justified. The final resolving team is never an escalation input.

All supplied incident links associate tickets with incidents that had already started. They are not future incident labels. `incident_reference.csv` retains summary plus the `What happened` section, excluding fixes and follow-up. Incident publication times are unknown, so `available_at` is blank and `availability_confirmed` is false by default. Do not substitute incident start time for publication time. If you have a verified date when the whole corpus was available, pass `--incidents-available-at`; it must be at least the latest incident start. Retrieval must still filter availability strictly before each ticket's creation. A post-May corpus availability means none can inform these May tickets.

## Cleaning details

- Header mapping, whitespace/Unicode cleanup, typed timestamps/numbers/booleans and normalized IDs.
- Primary-key, foreign-key, client, count, latest escalation and timestamp checks.
- Exact duplicate rows are removed; conflicting duplicate keys stop execution.
- Original workbooks remain unchanged. The 15 known Japanese templates are translated in cleaned historical descriptions and model text; translations do not use labels.
- Record IDs, support link notes, signatures and variable client/portfolio/instrument/fund identifiers are removed or generalized in model text.
- Blank descriptions fall back to titles; blank titles use the description. One case containing only a support note is excluded from text-model splits.
- Ten missing category/subcategory targets stay missing. No model-generated labels are substituted.
- Missing escalation assignment groups are filled from their linked tickets according to the supplied schema and logged. They are not model inputs.
- Escalations after resolution but before closure are retained and flagged. Manager approval does not change labels.
- Unknown escalation outcomes remain blank, not zero.

See `prepared_data/PREPARATION_REPORT.md`, `preparation_report.json`, `feature_policy.json` and `audit/` for counts, exclusions and assumptions.

## Optional dates

```sh
python prepare_data.py --train-end 2024-05-22 --validation-end 2024-05-26 --as-of "2024-10-12 01:18:55"
```

By default the creation-date boundaries are selected near the 70th and 85th percentile days. The observation cutoff is the maximum timestamp in the files, documented as an inferred synthetic-export cutoff. Supplying a cutoff does not prove the source covers all relevant events.

The generated data is synthetic. High scores, repeated templates and the unusually high escalation rate cannot establish real-world performance. No trained model, embeddings or TF-IDF vocabulary are produced during preparation.
