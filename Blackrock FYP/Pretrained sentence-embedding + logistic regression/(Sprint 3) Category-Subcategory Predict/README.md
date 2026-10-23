# Category and subcategory pair prediction

One multiclass logistic regression predicts a JSON-encoded [category, subcategory] pair from frozen all-MiniLM-L6-v2 sentence embeddings. Only prepared short description + description (`model_text`) are inputs. Category, subcategory, priority, team, escalation/incident outcomes and IDs are not predictor features. IDs are used for joins and grouping only. This is a separate model from escalation prediction.

## Run in VS Code
Open the extracted `ticket_data_preparation` folder. In its integrated terminal:

```powershell
python -m pip install -r category_pair_model/requirements.txt
python category_pair_model/train_category_pair.py
python -m unittest discover -s category_pair_model -p "test_category_pair.py" -v
python category_pair_model/predict_category_pair.py
```

Or open `Category_Subcategory_Pair.ipynb`, select the Python environment and Run All. The encoder is bundled in `model_approach3/encoder`; no model download is needed. The notebook reruns training and regenerates results. Input CSV for prediction can contain `model_text` (preferred: same cleaned initial text as training), or `short_description` + `description`. Raw input must not contain later resolution or incident support notes. Example command:

```powershell
python category_pair_model/predict_category_pair.py --input new_tickets.csv --output pair_predictions.csv
```

## Grouped evaluation only
The new model does not read chronological split files. It joins `prepared_data/ticket_features.csv` and `ticket_targets.csv` one-to-one, keeping usable text and both reviewed labels. Unknown escalation outcomes do not exclude a ticket with valid category labels. Dates do not split the new experiment, and tickets excluded only for crossing chronological boundaries can be used here.

Identical descriptions and the existing incident/problem event groups are merged into connected components. Five outer StratifiedGroupKFold folds produce exactly one held-out prediction per eligible ticket. In each outer training set, three inner grouped folds choose C from 0.01, 0.1, 1, 10, 100 by pooled pair macro F1 (ties prefer smaller C). The outer test fold is untouched by selection. Stratification is approximate because groups cannot be broken apart. Frozen encoder weights are not fitted on these tickets. No alert threshold or 90% recall target applies: prediction chooses the highest-scoring pair.

After evaluation, a separate grouped C selection on all eligible data precedes the final deployment fit. Reported evaluation comes exclusively from outer held-out predictions, not this final fit. Pairs absent from an outer training fold remain in its test set and count as errors if missed. The model cannot invent unseen category/subcategory pairs.

## Outputs
- `grouped_metrics.csv`: pooled pair/category/subcategory accuracy, macro F1 and weighted F1, plus a training-majority baseline.
- `grouped_fold_metrics.csv`: individual fold results and unseen-pair counts.
- `grouped_oof_predictions.csv`: one held-out prediction per ticket.
- `misclassified_pairs.csv`: errors for analysis.
- `*_confusion_matrix.csv` and `*_classification_report.json`: detailed class evaluation.
- `outer_fold_assignments.csv`, `inner_model_selection.csv`, `final_model_selection.csv`: split and tuning audits.
- `prepared_pairs.csv`, `excluded_rows.csv`, `pair_support.csv`: eligible data, exclusions and label/group coverage.
- `category_pair_model.joblib`: final classifier and encoder configuration.
- `run_manifest.json`: configuration and limitations.

Pair accuracy requires BOTH fields to match. Category accuracy can still be correct when subcategory is wrong. Macro F1 weights each evaluated class equally. Pair confidence is an uncalibrated model score, not a guaranteed correctness probability.

These are synthetic data and retrospective reviewed labels, with initial text availability assumed by the existing preparation. Results do not establish real-world performance or improvement over the existing 70–80% category assignment system. Grouped testing measures generalization across held-out groups, not temporal drift. Existing escalation scripts and notebooks retain their original evaluations; the grouped-only requirement applies to this new pair model.

## Added experiment: semantic template-family proxy

Run the added section in the SAME notebook, or:

```powershell
python category_pair_model/train_template_family.py
python -m unittest discover -s category_pair_model -p "test_*.py" -v
```

The default cosine threshold is 0.90, an exploratory starting rule fixed before evaluating classification accuracy. It is not a validated universal threshold. The frozen MiniLM encoder compares distinct cleaned texts; any pair above threshold is linked, and these links are combined with all original exact-text/event groups. Connected components become the group IDs for the SAME five outer / three inner grouped CV design. Labels do not determine groups. Original results remain under `results`; added results go under `results_template_family`.

Review `threshold_review_examples.csv` and `similarity_links.csv` to decide whether the rule captures actual paraphrases. Connected components are transitive: A can join C through B even if A/C similarity is below threshold. Event links can further enlarge groups. Check `template_group_sizes.csv`, `grouping_diagnostics.json`, per-pair group support, fold sizes and unseen-pair counts. Similarity can combine distinct issues, or fail to combine real paraphrases: these are semantic clusters, not proven original template families. If source template IDs become available, prefer those.

To use a different threshold after a label-blind review, run `python category_pair_model/train_template_family.py --threshold 0.92`. This overwrites the added experiment's output folder. Do not select a threshold based on test accuracy; any repeated test-informed experiments are exploratory. Grouping is a retrospective evaluation design using all text and frozen embeddings, not a trained production classifier or a chronological test.

`comparison_with_original.csv` compares pooled held-out scores. Its samples match the original pair task, but fold assignments differ intentionally. A lower result may reflect unseen pair labels in training or overly broad groups, so inspect coverage before attributing the entire change to template leakage. The additional final classifier is saved in `results_template_family/category_pair_model.joblib`; the original inference script still defaults to the original model.

Use the added model for new-ticket prediction explicitly:

```powershell
python category_pair_model/predict_category_pair.py --model category_pair_model/results_template_family/category_pair_model.joblib
```

## Third experiment: template-only grouping
Run the final section in the same notebook, or `python category_pair_model/train_template_only.py` from the project root. It retains the 0.90 cosine cutoff and transitive connected components, but ignores incident/problem links. Exact duplicates and all similarity edges remain within folds. Five outer and three inner grouped folds select/evaluate the same embeddings + LR model on the same eligible rows.

Results are saved separately in `results_template_only`. `comparison_all_groupings.csv` compares all three experiments; `event_cross_fold_audit.csv` shows event groups that cross folds by design. This evaluates held-out wording families, not unseen incidents. It is still an approximate family definition, not the stricter all-pairs rule. Synthetic templates and repeated experimentation limit real-world interpretation. No cutoff is selected based on test scores.

For prediction with this final model:
```powershell
python category_pair_model/predict_category_pair.py --model category_pair_model/results_template_only/category_pair_model.joblib
```
