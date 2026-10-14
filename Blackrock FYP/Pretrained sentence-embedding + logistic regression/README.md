# Approach 3: executed semantic escalation model

Open Approach_3_Escalation.ipynb to see the completed outputs without rerunning. Read MODEL_EVALUATION_REPORT.md for the follow-up: the updated model alerts on 386/458 test tickets, with 54 false positives and 54 missed escalations. The original model is retained for comparison.

## VS Code
1. Extract the full ZIP. Open the ticket_data_preparation folder in VS Code.
2. Install the Python and Jupyter VS Code extensions if needed.
3. Use a Python 3.12 virtual environment and run in the VS Code terminal:

    python -m pip install -r model_approach3/requirements.txt

4. Open model_approach3/Approach_3_Escalation.ipynb, select that environment as the kernel, and Run All. The notebook assumes its containing directory as its working directory (VS Code's usual notebook behavior).

The frozen encoder is bundled. Installing dependencies still requires internet access. requirements.txt records exact versions from the completed run. The run used CPU only. Saved joblib files should only be loaded from trusted sources.

## New predictions
Prepare the same model_text field using the original data-preparation cleaning first. From model_approach3:

    python predict.py prepared_tickets.csv --output new_predictions.csv

Input requires nonempty model_text and optionally unique ticket_id. Predictions use only model_text even if additional columns are present. No fresh labels or incident references are required to score a ticket. These synthetic-data scores are not validated real-world risk probabilities.

The original preparation files remain unchanged. model_approach3/data contains exact copies of prepared_data/splits/escalation. Results can be reproduced from the notebook; the frozen encoder is not fine-tuned. The original strict_splits are retained but not used in this prototype experiment.

The final notebook section saves the reduced-false-positive threshold as the default. Run All through that section. Original result files remain unchanged; updated evaluation filenames use reduced_fp or false_positive_comparison.

## Latest: structured feature experiment
The final notebook section adds category, subcategory and assumed initial priority to text embeddings. Run all cells to reproduce. Use `python predict_structured.py prepared_tickets.csv` to score with this model. The input requires model_text, category, subcategory and priority (optionally unique ticket_id). The original predict.py continues using the text-only model. See the latest section of MODEL_EVALUATION_REPORT.md.

## Grouped evaluation
The final notebook section performs five outer folds and three inner grouped folds. It compares both models with no shared exact description or event group across outer training and held-out folds. Results use the grouped_ prefix. This is exploratory grouped generalization, not a fresh chronological test. It does not overwrite deployment models.
