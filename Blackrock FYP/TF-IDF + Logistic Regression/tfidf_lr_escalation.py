"""
TF-IDF + Logistic Regression baseline for escalation prediction,
built to be directly comparable to Wei Kang's sentence-embedding + LR results.

Reuses the shared ../prepared_data/splits/escalation/{train,validation,test}.csv
(already cleaned, leakage-checked, same chronological boundaries) produced by
../prepare_data.py -- no separate data preparation here.

Two experiments, matching MODEL_EVALUATION_REPORT.md's structure:
  1. Text only (TF-IDF on model_text)
  2. Text + structured fields (category, subcategory, priority)

Run from this folder: `python tfidf_lr_escalation.py`
Assumes the repo layout as checked in (relative paths to sibling folders).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # .../Blackrock FYP

SPLIT_DIR = REPO_ROOT / "prepared_data" / "splits" / "escalation"
FIELDS_PATH = (
    REPO_ROOT
    / "Pretrained sentence-embedding + logistic regression"
    / "data"
    / "assumed_initial_fields.csv"
)
OUT_DIR = HERE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_GRID = [0.01, 0.1, 1, 10, 100]


def load_splits():
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    val = pd.read_csv(SPLIT_DIR / "validation.csv")
    test = pd.read_csv(SPLIT_DIR / "test.csv")
    return train, val, test


def select_C(X_train, y_train, X_val, y_val):
    """Pick C by validation average precision, same selection rule WK used."""
    best_c, best_ap, best_model = None, -1, None
    for c in C_GRID:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
        clf.fit(X_train, y_train)
        val_scores = clf.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, val_scores)
        if ap > best_ap:
            best_c, best_ap, best_model = c, ap, clf
    return best_c, best_ap, best_model


def select_threshold(y_val, val_scores, min_recall=0.90):
    """Minimize false positives on validation subject to recall >= min_recall.
    Ties favor recall, then threshold -- same rule WK documented."""
    candidates = np.unique(val_scores)
    best = None  # (fp, -recall, threshold)
    for t in candidates:
        preds = (val_scores >= t).astype(int)
        recall = recall_score(y_val, preds, zero_division=0)
        if recall < min_recall:
            continue
        fp = int(((preds == 1) & (y_val == 0)).sum())
        key = (fp, -recall, t)
        if best is None or key < best[0]:
            best = (key, t, recall, fp)
    if best is None:
        t = candidates.min()
        preds = (val_scores >= t).astype(int)
        recall = recall_score(y_val, preds, zero_division=0)
        fp = int(((preds == 1) & (y_val == 0)).sum())
        return t, recall, fp
    return best[1], best[2], best[3]


def evaluate(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
        "false_positives": int(fp),
        "missed_escalations": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "threshold": float(threshold),
    }


def run_experiment(name, X_train, X_val, X_test, y_train, y_val, y_test):
    c, val_ap, clf = select_C(X_train, y_train, X_val, y_val)
    val_scores = clf.predict_proba(X_val)[:, 1]
    threshold, achieved_val_recall, val_fp = select_threshold(y_val, val_scores)

    test_scores = clf.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_scores, threshold)
    val_metrics = evaluate(y_val, val_scores, threshold)

    result = {
        "experiment": name,
        "selected_C": c,
        "validation_average_precision_at_selection": val_ap,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    print(f"\n=== {name} ===")
    print(f"Selected C={c}, threshold={threshold:.4f}")
    print(f"Validation: recall={val_metrics['recall']:.4f} FP={val_metrics['false_positives']}")
    print(f"Test: precision={test_metrics['precision']:.4f} recall={test_metrics['recall']:.4f} "
          f"f1={test_metrics['f1']:.4f} roc_auc={test_metrics['roc_auc']:.4f} "
          f"FP={test_metrics['false_positives']} missed={test_metrics['missed_escalations']}")
    return result, clf


def main():
    train, val, test = load_splits()
    y_train, y_val, y_test = train["escalation_label"], val["escalation_label"], test["escalation_label"]

    # ---------- Experiment 1: text only ----------
    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)
    X_train_text = vectorizer.fit_transform(train["model_text"])
    X_val_text = vectorizer.transform(val["model_text"])
    X_test_text = vectorizer.transform(test["model_text"])

    result_text, clf_text = run_experiment(
        "TF-IDF text only", X_train_text, X_val_text, X_test_text, y_train, y_val, y_test
    )

    # ---------- Experiment 2: text + structured fields ----------
    fields = pd.read_csv(FIELDS_PATH)
    train_f = train.merge(fields, on="ticket_id", how="left")
    val_f = val.merge(fields, on="ticket_id", how="left")
    test_f = test.merge(fields, on="ticket_id", how="left")

    struct_cols = ["category", "subcategory", "priority"]
    for df in (train_f, val_f, test_f):
        df[struct_cols] = df[struct_cols].fillna("__MISSING__")

    ohe = OneHotEncoder(handle_unknown="ignore")
    X_train_struct = ohe.fit_transform(train_f[struct_cols])
    X_val_struct = ohe.transform(val_f[struct_cols])
    X_test_struct = ohe.transform(test_f[struct_cols])

    X_train_combo = hstack([X_train_text, X_train_struct], format="csr")
    X_val_combo = hstack([X_val_text, X_val_struct], format="csr")
    X_test_combo = hstack([X_test_text, X_test_struct], format="csr")

    result_combo, clf_combo = run_experiment(
        "TF-IDF text + structured", X_train_combo, X_val_combo, X_test_combo, y_train, y_val, y_test
    )

    # ---------- Save results ----------
    summary = {"text_only": result_text, "text_plus_structured": result_combo}
    with open(OUT_DIR / "tfidf_lr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Comparison table vs Wei Kang's reported chronological test numbers
    wk_text_only = {"precision": 0.8601, "recall": 0.8601, "f1": 0.8601, "roc_auc": 0.6508,
                     "false_positives": 54, "missed_escalations": 54}
    wk_structured = {"precision": 0.8670, "recall": 0.8782, "f1": 0.8726, "roc_auc": 0.7212,
                      "false_positives": 52, "missed_escalations": 47}

    rows = []
    for label, ours, wk in [
        ("Text only", result_text["test_metrics"], wk_text_only),
        ("Text + structured", result_combo["test_metrics"], wk_structured),
    ]:
        rows.append({
            "model": label,
            "tfidf_lr_precision": round(ours["precision"], 4),
            "wk_embedding_lr_precision": wk["precision"],
            "tfidf_lr_recall": round(ours["recall"], 4),
            "wk_embedding_lr_recall": wk["recall"],
            "tfidf_lr_f1": round(ours["f1"], 4),
            "wk_embedding_lr_f1": wk["f1"],
            "tfidf_lr_roc_auc": round(ours["roc_auc"], 4),
            "wk_embedding_lr_roc_auc": wk["roc_auc"],
            "tfidf_lr_false_positives": ours["false_positives"],
            "wk_false_positives": wk["false_positives"],
            "tfidf_lr_missed_escalations": ours["missed_escalations"],
            "wk_missed_escalations": wk["missed_escalations"],
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT_DIR / "comparison_vs_wk.csv", index=False)
    print("\n=== Comparison vs Wei Kang's sentence-embedding + LR (chronological test) ===")
    print(comparison.to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'tfidf_lr_summary.json'}")
    print(f"Saved: {OUT_DIR / 'comparison_vs_wk.csv'}")


if __name__ == "__main__":
    main()
