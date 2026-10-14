# Latest: nested grouped evaluation

Executed five outer StratifiedGroupKFold folds with three inner grouped folds for model and threshold selection. All 2,596 previously eligible tickets are evaluated once out-of-fold per model (2,219 escalated; 377 non-escalated). No previously excluded tickets were restored.

811 connected groups keep together shared cleaned model_text or event_group_id, including transitive links. Largest group: 214 tickets. Outer folds contain 518–522 tickets and 75–76 non-escalations each. Assertions verified no exact-text or event-group overlap across outer train/held-out partitions and disjoint inner groups.

Within each outer training fold, C is selected from [0.01,0.1,1,10,100] by inner out-of-fold average precision. Threshold minimizes inner out-of-fold false positives subject to recall >=90%; ties favor recall then threshold. Each fitted one-hot encoder sees only its current training fold. Frozen pretrained text embeddings are reused. Outer labels are used only for evaluation. All classifier fits converged.

## Pooled held-out predictions
| Metric | Text only | Text + category/subcategory/assumed initial priority |
|---|---:|---:|
| False positives | 272 | 250 |
| Missed escalations | 270 | 250 |
| Correctly detected escalations | 1,949 | 1,969 |
| Correctly rejected non-escalations | 105 | 127 |
| Precision | 87.75% | 88.73% |
| Recall | 87.83% | 88.73% |
| F1 | 87.79% | 88.73% |
| Accuracy | 79.12% | 80.74% |
| Balanced accuracy | 57.84% | 61.21% |
| Average precision | 92.53% | 94.96% |
| ROC AUC | 0.7055 | 0.7711 |
| Alert rate | 85.55% | 85.48% |

Structured features produce 22 fewer false positives (8.09% reduction) and 20 fewer missed escalations (7.41% reduction), with almost the same alert rate. Still, 250/377 (66.31%) actual non-escalations are flagged. Always alerting has 377 false positives and no missed escalations, with F1 92.17%; this illustrates why F1 alone is insufficient for the selective-alert objective.

Pooled scores come from five different models/thresholds. Per-fold mean ROC AUC is 0.7085 for text only and 0.7733 for structured inputs. Structured-model recall varies from 80.27% to 93.23% across folds; mean 88.74%, standard deviation 5.24 percentage points. These are fold summaries, not independent confidence intervals. Inner recall >=90% does not guarantee outer recall >=90%.

## Interpretation and limits
This is evidence of a modest exploratory benefit from the added fields under group-held-out evaluation, not proof of production performance. It removes exact-text/event-group overlap; paraphrases may still cross groups. Category/subcategory/priority are assumed correct initial values. Synthetic data, unverified snapshots, retrospective outcomes and repeated examination of this dataset remain limitations.

Dates are mixed by design. This tests unseen groups, not future chronological deployment. Do not compare pooled error counts directly with the previous 458-ticket test; the evaluation set and question differ. No deployment model was selected or overwritten by cross-validation. Previously saved models remain from the earlier experiments.

## New outputs
- results/grouped_pooled_metrics.csv: pooled held-out metrics.
- results/grouped_fold_metrics.csv and grouped_fold_variability.csv: per-fold results and spread.
- results/grouped_oof_predictions.csv: one held-out prediction per ticket per model.
- results/grouped_fold_assignments.csv and grouped_fold_summary.csv: auditable split membership.
- results/grouped_inner_model_selection.csv: training-only model selection.
- results/grouped_confusion_*.csv and grouped_classification_*.json: confusion matrices and classification reports.
- results/grouped_confusion_comparison.png: comparison chart.
- results/grouped_run_manifest.json: grouping and experiment configuration.

---

# Latest experiment: text + category + subcategory + assumed initial priority

Added to the same executed Approach_3_Escalation.ipynb. The original text-only experiments are preserved.

Inputs: cleaned short description + description (384 frozen semantic embedding dimensions), plus training-fitted one-hot category, subcategory and priority (55 dimensions), totaling 439 inputs. Missing values use __MISSING__; unseen categories are accepted. No client identity, team assignment or outcomes are predictors.

**Assumption:** the supplied structured fields are correct initial-ticket values. They are not verified initial snapshots. Results are conditional on this assumption.

C=1.0 selected by validation average precision from five candidates. Threshold=0.7031941591423143 selected to minimize validation false positives while maintaining at least 90% recall. Validation recall is 90.31%, FP=38, FN=28. Threshold selection uses no test labels.

| Test metric | Previous text-only, recall-target threshold | Text + structured fields |
|---|---:|---:|
| False positives | 54 | 52 |
| Missed escalations | 54 | 47 |
| Correctly detected escalations | 332 | 339 |
| Correctly rejected non-escalations | 18 | 20 |
| Precision | 86.01% | 86.70% |
| Recall | 86.01% | 87.82% |
| F1 | 86.01% | 87.26% |
| Accuracy | 76.42% | 78.38% |
| Balanced accuracy | 55.51% | 57.80% |
| Average precision | 90.65% | 92.88% |
| ROC AUC | 0.6508 | 0.7212 |
| Alert rate | 84.28% | 85.37% |

Interpretation: modest improvement in this exploratory comparison. Two fewer false alarms and seven fewer missed escalations, with five additional total alerts (391 vs 386). Ranking improves, but 52 of 72 non-escalations still trigger alerts. Test recall does not reach the provisional 90% validation target. The improvement is not proven statistically or on independent real-world data. This experiment cannot determine which individual structured field caused the gain; that would require ablation comparisons.

The test set has been examined repeatedly; these are descriptive follow-up results. Synthetic records, unverified initial snapshots, retrospective labels and repeated templates remain limitations. A fresh representative holdout is needed before selecting a production system.

## Latest files
- results/escalation_model_structured.joblib: new classifier, fitted one-hot encoder, threshold and metadata. Previous models are retained separately.
- results/structured_comparison.csv: before/after validation and test metrics.
- results/*_predictions_structured.csv: updated risk scores and predictions.
- results/*_confusion_matrix_structured.csv and *_classification_report_structured.json: evaluation.
- results/structured_confusion_comparison.png: comparison chart.
- results/structured_validation_model_selection.csv and structured_validation_thresholds.csv: selection audit.
- data/assumed_initial_fields.csv: four-column source mapping (ID and the three structured fields).
- predict_structured.py: score cleaned model_text plus category, subcategory and priority using the new model.

Checks passed: one-to-one aligned joins, train-only encoding, convergence, saved-model score reproduction, unseen category handling, notebook output validation. Original preparation ZIP members remain unchanged.

---

# Updated operating threshold — fewer false positives

The notebook now contains a completed follow-up section, **Follow-up: reduce false positives while preserving escalation recall**. Earlier sections retain the original experiment. The saved default model uses the updated threshold.

The encoder and classifier are unchanged. The decision threshold increased from 0.630039 to 0.713439, selected using validation only to minimize false positives while retaining at least 90% validation escalation recall. Ties favor recall, then threshold. This is an operating-point change, not improved text understanding or ranking.

| Test metric | Original | Updated |
|---|---:|---:|
| False positives | 71 | 54 |
| True negatives | 1 | 18 |
| True positives | 386 | 332 |
| Missed escalations | 0 | 54 |
| Precision | 84.46% | 86.01% |
| Recall | 100.00% | 86.01% |
| F1 | 91.58% | 86.01% |
| Accuracy | 84.50% | 76.42% |
| Balanced accuracy | 50.69% | 55.51% |
| Alert rate | 99.78% | 84.28% |
| Average precision | 90.65% | 90.65% |
| ROC AUC | 0.6508 | 0.6508 |

False positives fell by 17 (23.94%). Alerts fell from 457 to 386. Of those 71 removed alerts, 17 were false alarms and 54 were real escalations. Thus this is not an unqualified overall improvement: reduced false alarms come at a substantial recall cost.

Validation false positives fell from 52 to 40, with recall 91.35%. The 90% validation target was satisfied but test recall was only 86.01%; the target is not a guarantee on unseen data. The classifier's ranking scores and ranking metrics are unchanged.

The test set was already examined in the original experiment. These are follow-up descriptive results, not a new unbiased holdout evaluation. No test labels enter threshold selection. Confirm the operational recall requirement with stakeholders and use a fresh real-data holdout for final assessment.

## Updated artifacts
- `results/escalation_model.joblib`: same classifier, updated default threshold used by predict.py.
- `results/escalation_model_original.joblib`: original threshold preserved.
- `results/false_positive_comparison.csv`: validation and test before/after metrics.
- `results/*_predictions_reduced_fp.csv`: updated predictions.
- `results/*_confusion_matrix_reduced_fp.csv` and `*_classification_report_reduced_fp.json`: updated evaluation.
- `results/confusion_matrix_comparison.png`: before/after matrices.
- `results/false_positive_threshold_candidates.csv`: validation selection audit.
- `results/unseen_text_reduced_fp.csv`: secondary unseen-text evaluation.
- Original files without the reduced_fp suffix retain the original experiment results.

Saved-model reload and prediction-consistency checks passed. All original data-preparation ZIP members remain unchanged. Synthetic-data and retrospective-split limitations still apply.

---

# Original experiment (retained for reference)

# Approach 3 — completed model evaluation

## What was trained
Frozen sentence-transformers/all-MiniLM-L6-v2 (384-dimensional normalized embeddings) + logistic regression. Only cleaned model_text is used. No incident similarity, category, priority, team or outcome fields are inputs. All raised escalations count regardless of approval.

Training: 1,797 tickets (1,544 positive, 253 negative). Validation: 341 (289 positive, 52 negative). Test: 458 (386 positive, 72 negative). Unique descriptions across these partitions: 1,040. No descriptions exceeded the 256-token encoder limit.

Selected C=1.0 by validation average precision (0.924196), from [0.01, 0.1, 1, 10, 100]. Selected alert threshold=0.6300394366039118 by validation positive-class F1, with ties favoring precision. Model fitting used training only. Test was evaluated after model and threshold selection; no refit or test-driven tuning was performed.

## Test metrics
| Metric | Semantic model, selected threshold | Always alert |
|---|---:|---:|
| Accuracy | 84.50% | 84.28% |
| Balanced accuracy | 50.69% | 50.00% |
| Precision | 84.46% | 84.28% |
| Recall | 100.00% | 100.00% |
| F1 | 91.58% | 91.47% |
| Average precision | 90.65% | 84.28% |
| ROC AUC | 0.6508 | 0.5000 |
| Alert rate | 99.78% | 100.00% |
| Brier score (lower is better) | 0.1294 | 0.1328 |

The dummy ranking/probability score is the constant TRAINING prevalence; its alert threshold produces always-alert or always-negative decisions. Average precision of constant scores equals test prevalence.

## Test confusion matrix
Rows = actual, columns = predicted.

| | Predicted not escalated | Predicted escalated |
|---|---:|---:|
| Actually not escalated | 1 | 71 |
| Actually escalated | 0 | 386 |

## Interpretation
The model flags 457 of 458 tickets. It catches every test escalation but filters out only one non-escalation. The high positive-class F1 is largely explained by the high escalation prevalence. The selected threshold maximized validation F1 by alerting on ALL validation tickets; this policy offers almost no reduction in review workload. At threshold 0.5 the model alerts on every test ticket.

Ranking has some signal: average precision exceeds the no-skill prevalence and ROC AUC is 0.651. This does not make the current operating point useful for selective alerts. Before a future evaluation, specify an acceptable alert budget or required recall and select a threshold on validation accordingly. Do not tune against the current test results.

On 97 test descriptions not seen verbatim in training or validation: precision 85.42%, recall 100%, F1 92.13%, average precision 90.13%, ROC AUC 0.6317, alert rate 98.97%. This subset contains 82 escalations and only 15 negatives; uncertainty is substantial and similar paraphrases may remain.

## Limitations
- Synthetic records cannot establish real-world predictive performance.
- Preparation uses retrospective labels known by the final observation cutoff, rather than a deployable historical training cutoff. Strict temporal validation in the source package was unusable.
- Initial snapshots were not verified. Removing obvious later notes does not prove remaining text was available at arrival.
- 361 of 458 test descriptions repeat earlier text exactly. The unseen-text diagnostic is smaller and not an independent test sample.
- Scores are not separately calibrated or validated as real-world risk probabilities.
- No statistical confidence interval or claim of significant improvement is made.
- The encoder was frozen; only the logistic regression classifier was trained.

## Delivered files
Approach_3_Escalation.ipynb is executed, with tables and charts. results/ includes metrics, validation selection, thresholds, confusion matrices, classification reports, predictions, embeddings, saved classifier, charts and run manifest. encoder/ bundles the pretrained model so reruns do not require downloading weights. data/ contains exact copies of the preparation escalation partitions. predict.py scores already-cleaned model_text. See README.md for VS Code instructions.

Checks passed: split IDs and groups disjoint, chronological order, valid text and binary labels, finite 384-dimensional embeddings, classifier convergence, saved classifier reproduces scores, saved predictions match the selected threshold. All original preparation ZIP members were retained unchanged in the updated package.
