# Data preparation report

Prepared from the three finalised synthetic workbooks. Original Excel inputs are retained unchanged.

## Cleaning and labels

- tickets: **2636**
- escalation events: **2630**
- incidents: **774**
- escalated tickets: **2240**
- non escalated tickets: **380**
- unknown tickets: **16**
- incident case links: **487**
- translated ticket descriptions: **15**
- removed support notes: **487**
- missing descriptions: **117**
- missing titles: **88**
- unusable model text: **1**
- missing category targets: **10**
- cross boundary event group tickets: **23**
- unique model texts: **1050**

Escalation label 1 means ANY raised escalation, regardless of manager approval. Label 0 requires closure without any event. Open/unresolved outcomes remain unknown. Multiple events are aggregated to one ticket; no ticket rows are multiplied by joins.

Positive label availability uses the first event observable in the export (the later of OPENED_AT and SYS_CREATED_ON). Negative availability uses Closed. Dates retain the source timezone convention; no timezone was invented.

Missing targets are not invented. Missing descriptions fall back to titles. Known Japanese templates are translated; original descriptions remain in input/; cleaned historical descriptions are translated to English. Model text removes support notes, record IDs, email signatures and variable client/portfolio/instrument/fund identifiers. Missing escalation assignment groups are filled from the documented linked-ticket field and logged.

## Retrospective prototype splits

Train creation < 2024-05-22 00:00:00; validation creation < 2024-05-26 00:00:00; test thereafter. Observation cutoff: 2024-10-12 01:18:55 (inferred from maximum observed timestamp unless overridden). These splits use all outcomes known by the final observation cutoff, including outcomes recorded after the historical training/validation boundaries. They are suitable for a retrospective synthetic prototype, NOT a strict prospective backtest. Incident/problem groups crossing partitions are excluded from all splits.

| Task | Train | Validation | Test |
|---|---:|---:|---:|
| escalation | 1797 | 341 | 458 |
| category | 1803 | 338 | 461 |
| subcategory | 1803 | 338 | 461 |
| priority | 1810 | 341 | 461 |
| team | 1777 | 336 | 451 |

## Strict timing diagnostic splits

strict_splits/ requires outcomes to be observable by each historical partition cutoff. This reveals the short follow-up problem; do not fit or select a model on an empty or single-class validation set.

| Task | Train | Validation | Test |
|---|---:|---:|---:|
| escalation | 1503 | 207 | 458 |
| category | 817 | 0 | 461 |
| subcategory | 817 | 0 | 461 |
| priority | 820 | 0 | 461 |
| team | 802 | 0 | 451 |

Full class counts are in audit/class_distribution.csv. Small or missing classes are a limitation, not a reason to shuffle future data into training. The optional *_unseen_text.csv files exclude texts seen in earlier primary partitions and may be small or empty.

## Data quality findings

- raised_after_resolution_before_closure: 14
- empty_model_text: 1

## Important limitations

- Synthetic data; performance does not establish production accuracy.
- Only one month of ticket arrivals; class balance is not representative by assumption.
- Primary splits are RETROSPECTIVE. They use labels known by the final as-of and cannot establish what could have been trained or tuned at the original boundaries. strict_splits enforce that stronger timing rule, but are not viable for all tasks with this short dataset.
- No verified initial text/field history. Default models should use model_text only and acknowledge the snapshot assumption.
- Category, priority and team are reviewed targets. Their latest Updated time is used conservatively as a label-availability proxy. Unresolved team labels are excluded.
- No incident publication timestamps. Default incident reference rows are not eligible for time-safe retrieval until availability is supplied.
- Incident links are associations with prior incidents, not future-incident outcomes.
- Time split is purged of incident/problem groups spanning boundaries. Repeated synthetic texts remain; unseen_text subsets are supplied separately.
- Do not concatenate split files, include targets as features, or fit text transforms on validation/test data.

## Files to use

- Start with splits/escalation/train.csv, validation.csv and test.csv. Use model_text as X and escalation_label as y.
- Use the corresponding category, subcategory, priority or team folders for those tasks. Identifier/date/group columns are metadata, not model inputs.
- ticket_features.csv and ticket_targets.csv are master preparation tables, not a combined training set.
- cleaned/ retains normalized historical fields for traceability, not unrestricted model inputs.
- incident_reference.csv contains symptom-only text; incident_case_links.csv is an audit/grouping table.
- audit/excluded_rows.csv explains each task-specific exclusion. Other audit files record missingness, distributions and changes.
- feature_policy.json defines allowed and forbidden inputs.
- Fit TF-IDF, scaling, encoders and any resampling only after selecting training rows. No model fitting is performed by this preparation script.
