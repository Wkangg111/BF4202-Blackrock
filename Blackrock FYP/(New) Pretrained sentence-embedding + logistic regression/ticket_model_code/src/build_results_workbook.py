"""Builds artifacts/test_results.xlsx from the CSV tables exported by the notebook (artifacts/results/)."""
import sys
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "artifacts/results"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else RES.parent / "test_results.xlsx"
T = {p.stem: pd.read_csv(p) for p in RES.glob("*.csv")}

FONT, BOLD = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
TITLE, NOTE = Font(name="Arial", size=13, bold=True), Font(name="Arial", size=9, italic=True, color="52514E")
HEAD_FILL = PatternFill("solid", fgColor="DCE8F7")
THIN = Border(bottom=Side(style="thin", color="B7B6B0"))
GROUPINGS = ["random", "client", "short_desc", "template", "semantic_k200", "semantic_k50"]

S1_LABELS = {"grouping": "Grouping", "fold": "Fold", "features": "Inputs", "AUC": "ROC-AUC", "PR_AUC": "PR-AUC",
             "Brier": "Brier score", "base_rate": "Escalation base rate", "C": "Chosen C",
             "coverage": "Share answered 0 or 1", "pred_1_share": "Share answered 1",
             "precision_1": "Precision of '1'", "recall_1": "Escalations flagged '1' (recall)",
             "NPV_0": "Accuracy of '0' (NPV)", "pred_0_share": "Share answered 0",
             "OOD_abstain": "Refused: unlike training data", "near_dup_in_train": "Test tickets with near-copy in training",
             "t_lo": "Threshold: say 0 below", "t_hi": "Threshold: say 1 at/above"}
S2_LABELS = {"grouping": "Grouping", "fold": "Fold", "pair_acc_forced": "Pair accuracy (always answering)",
             "cat_acc_forced": "Category accuracy (always answering)", "pair_macroF1_forced": "Pair macro-F1 (always answering)",
             "C": "Chosen C", "answer_pair": "Share answered with pair",
             "pair_acc_when_answered": "Pair accuracy when pair given", "answer_cat_only": "Share answered category only",
             "cat_acc_when_cat_only": "Category accuracy when category only", "abstain": "Share abstained",
             "OOD_abstain": "Refused: unlike training data", "unseen_label_rows": "Test tickets with pair never seen in training",
             "unseen_given_pair": "Unseen-pair tickets given a pair (wrong)",
             "unseen_safe": "Unseen-pair tickets handled safely", "near_dup_in_train": "Test tickets with near-copy in training",
             "t_pair": "Threshold: pair confidence", "t_cat": "Threshold: category confidence", "tau": "Similarity cut-off"}
PCT = {"Escalation base rate", "Share answered 0 or 1", "Share answered 1", "Precision of '1'", "Escalations flagged '1' (recall)",
       "Accuracy of '0' (NPV)", "Share answered 0", "Refused: unlike training data", "Test tickets with near-copy in training",
       "Pair accuracy (always answering)", "Category accuracy (always answering)", "Share answered with pair",
       "Pair accuracy when pair given", "Share answered category only", "Category accuracy when category only",
       "Share abstained", "Test tickets with pair never seen in training", "Unseen-pair tickets given a pair (wrong)",
       "Unseen-pair tickets handled safely", "share", "observed escalation rate", "95% CI low", "95% CI high",
       "old observed rate", "rate this band promised (new test folds)", "Share of row"}

wb = Workbook()
wb.remove(wb.active)


def style_header(ws, row, ncol, start=1):
    for c in range(start, start + ncol):
        cell = ws.cell(row=row, column=c)
        cell.font, cell.fill, cell.border = BOLD, HEAD_FILL, THIN
        cell.alignment = Alignment(wrap_text=True, vertical="center")


def fmt_cell(cell, header):
    cell.font = FONT
    if isinstance(cell.value, float) or (isinstance(cell.value, str) and cell.value.startswith("=")):
        if header in PCT:
            cell.number_format = "0.0%"
        elif header in ("Chosen C",):
            cell.number_format = "0.000"
        else:
            cell.number_format = "0.000"
    if isinstance(cell.value, (int,)) and header not in ("Fold",):
        cell.number_format = "#,##0"


def widths(ws, minw=10, maxw=48):
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        longest = max((len(str(c.value)) for c in col if c.value is not None and not str(c.value).startswith("=")), default=8)
        ws.column_dimensions[letter].width = max(minw, min(maxw, longest + 2))


def sheet_from_df(name, df, title, note, labels=None, start_row=4):
    ws = wb.create_sheet(name)
    ws["A1"], ws["A1"].font = title, TITLE
    ws["A2"], ws["A2"].font = note, NOTE
    df = df.rename(columns=labels or {})
    for j, h in enumerate(df.columns, 1):
        ws.cell(row=start_row, column=j, value=h)
    style_header(ws, start_row, len(df.columns))
    for i, row in enumerate(df.itertuples(index=False), start_row + 1):
        for j, v in enumerate(row, 1):
            if pd.isna(v):
                v = None
            elif hasattr(v, "item"):
                v = v.item()
            c = ws.cell(row=i, column=j, value=v)
            fmt_cell(c, df.columns[j - 1])
    ws.freeze_panes = ws.cell(row=start_row + 1, column=1)
    widths(ws)
    return ws, df


# ---------------- README ----------------
ws = wb.create_sheet("README")
info = T["run_info"].set_index("item")["value"]
lines = [
    ("Ticket intake model: all test results", TITLE),
    ("Synthetic data. Results show the pipeline works on the generator's patterns; they are not estimates of real-world performance.", NOTE),
    ("", FONT),
    ("How the testing worked", BOLD),
    (f"• Built and tested on the NEW case file only ({int(info['new cases']):,} cases). The OLD file ({int(info['old cases']):,} cases) is used only as an unseen stress test, with no training or tuning.", FONT),
    (f"• Nested cross-validation: {info['outer folds']} outer folds for testing; inside each, {info['inner folds']} inner folds choose C ({info['C grid']}) and every refusal threshold. Test folds are never used for tuning.", FONT),
    ("• Six grouping systems decide which tickets must stay together in one fold, so near-copies cannot leak from training into testing:", FONT),
    ("     random = no grouping (leakage baseline) · client = same client · short_desc = same short description · template = same wording with IDs/dates/numbers masked (primary) · semantic_k200 / semantic_k50 = same embedding cluster (k50 holds out whole topics; hardest).", FONT),
    (f"• Refusal targets: say '1' only where validated precision ≥ {float(info['stage 1 precision target']):.0%}; say '0' only where validated NPV ≥ {float(info['stage 1 NPV target']):.0%}; give a category/pair only where validated accuracy ≥ {float(info['stage 2 accuracy target']):.0%}; refuse any ticket less similar to training than the {info['OOD percentile']}st percentile seen on validation.", FONT),
    ("", FONT),
    ("Sheets", BOLD),
    ("Summary — mean and standard deviation across the 5 outer test folds for every grouping (live formulas over the fold sheets).", FONT),
    ("S1 folds — stage 1 (escalation) result for every grouping × outer fold.", FONT),
    ("S1 ablation — stage 1 with different inputs (priority/tier only, text only, both), template grouping, per fold + summary.", FONT),
    ("S1 decisions — every new ticket's 0 / 1 / ABSTAIN decision on its test fold vs what actually happened (template grouping).", FONT),
    ("S1 risk bands — escalation rate actually observed per risk band on the outer test folds, with 95% Wilson interval.", FONT),
    ("S2 folds — stage 2 (category/sub-category) result for every grouping × outer fold.", FONT),
    ("S2 confusions — most frequent wrong pairs when the model is forced to answer (template grouping).", FONT),
    ("Temporal — train Jan–Apr 2024, test May–Jun 2024.", FONT),
    ("Old S1 decisions / Old S1 bands — final model run unchanged on the old file: decisions and bands vs truth.", FONT),
    ("Old S2 outcomes / Old S2 mappings — old-file category results by case, and the label mappings the model proposed (for human review; not counted as correct).", FONT),
    ("Examples — four illustrative tickets scored by the final model, including an off-topic one.", FONT),
    ("Model settings / Grouping systems / Run info — final thresholds, group counts and configuration.", FONT),
    ("", FONT),
    ("Key metrics", BOLD),
    ("ROC-AUC — chance an escalated ticket scores higher than a non-escalated one (0.5 = coin flip, 1 = perfect).", FONT),
    ("PR-AUC — ranking quality focused on the escalated minority; compare with the base rate (~10%).", FONT),
    ("Brier score — mean squared error of the probabilities (lower is better).", FONT),
    ("Precision of '1' — share of tickets answered '1' that did escalate. NPV of '0' — share answered '0' that did not escalate.", FONT),
    ("Recall — share of all escalations that were answered '1' in advance.", FONT),
    ("Pair accuracy (always answering) — accuracy if the model had to give a pair for every ticket. 'When pair given' — accuracy on the tickets it chose to answer.", FONT),
    ("Unseen-pair tickets given a pair — test tickets whose true pair never appeared in training, yet the model named a pair (necessarily wrong). Lower is better.", FONT),
    ("Unseen-pair tickets handled safely — of those, share that were refused or given only the correct category. Higher is better.", FONT),
    ("Test tickets with near-copy in training — share of test tickets with a training ticket at cosine similarity ≥ 0.98; shows how much each grouping prevents leakage.", FONT),
    ("", FONT),
    ("Blank cells mean the metric does not apply (e.g. no unseen pairs exist under that grouping, or the model never gave that answer).", NOTE),
]
for i, (text, f) in enumerate(lines, 1):
    c = ws.cell(row=i, column=1, value=text); c.font = f
    c.alignment = Alignment(wrap_text=True, vertical="top")
ws.column_dimensions["A"].width = 150

# ---------------- fold sheets ----------------
s1_order = ["grouping", "fold", "AUC", "PR_AUC", "Brier", "base_rate", "coverage", "pred_1_share", "precision_1", "recall_1",
            "pred_0_share", "NPV_0", "OOD_abstain", "near_dup_in_train", "C", "t_lo", "t_hi"]
ws1, s1 = sheet_from_df("S1 folds", T["stage1_folds"][s1_order], "Stage 1 (escalation): every grouping × outer test fold",
                        "Each row is one outer test fold (~5,900 tickets). Thresholds and C were chosen on that fold's training part only.", S1_LABELS)
s2_order = ["grouping", "fold", "pair_acc_forced", "cat_acc_forced", "pair_macroF1_forced", "answer_pair", "pair_acc_when_answered",
            "answer_cat_only", "cat_acc_when_cat_only", "abstain", "OOD_abstain", "unseen_label_rows", "unseen_given_pair",
            "unseen_safe", "near_dup_in_train", "C", "t_pair", "t_cat", "tau"]
ws2, s2 = sheet_from_df("S2 folds", T["stage2_folds"][s2_order], "Stage 2 (category / sub-category pair): every grouping × outer test fold",
                        "Each row is one outer test fold (~5,750 labelled tickets).", S2_LABELS)


def summary_block(ws, top, title, fold_ws, fold_df, metrics, key_col="Grouping", keys=GROUPINGS, last_row=None):
    """Mean and SD per key, as formulas over the fold sheet."""
    ws.cell(row=top, column=1, value=title).font = BOLD
    hdr = top + 1
    ws.cell(row=hdr, column=1, value="Metric")
    for j, k in enumerate(keys):
        ws.cell(row=hdr, column=2 + j, value=f"{k} mean")
        ws.cell(row=hdr, column=2 + len(keys) + j, value=f"{k} SD")
    style_header(ws, hdr, 1 + 2 * len(keys))
    n_last = last_row or (4 + len(fold_df))
    kcol = get_column_letter(list(fold_df.columns).index(key_col) + 1)
    sheet = f"'{fold_ws.title}'"
    krange = f"{sheet}!${kcol}$5:${kcol}${n_last}"
    for i, m in enumerate(metrics, hdr + 1):
        ws.cell(row=i, column=1, value=m).font = FONT
        vcol = get_column_letter(list(fold_df.columns).index(m) + 1)
        vrange = f"{sheet}!${vcol}$5:${vcol}${n_last}"
        for j, k in enumerate(keys):
            mean_cell = ws.cell(row=i, column=2 + j, value=f'=IFERROR(AVERAGEIFS({vrange},{krange},"{k}"),"n/a")')
            mref = mean_cell.coordinate
            sd = (f'=IFERROR(SQRT(SUMPRODUCT(({krange}="{k}")*({vrange}<>"")*({vrange}-{mref})^2)'
                  f'/COUNTIFS({krange},"{k}",{vrange},"<>")),"n/a")')
            sd_cell = ws.cell(row=i, column=2 + len(keys) + j, value=sd)
            for c in (mean_cell, sd_cell):
                c.font = FONT
                c.number_format = "0.0%" if m in PCT else "0.000"
                c.alignment = Alignment(horizontal="right")
    return hdr + len(metrics) + 2


ws = wb.create_sheet("Summary", 1)
ws["A1"], ws["A1"].font = "Summary: mean and standard deviation across the 5 outer test folds", TITLE
ws["A2"], ws["A2"].font = ("All values are formulas over the 'S1 folds' and 'S2 folds' sheets (SD = population SD across folds). "
                           "'n/a' = metric does not apply under that grouping."), NOTE
s1_metrics = ["ROC-AUC", "PR-AUC", "Brier score", "Share answered 0 or 1", "Share answered 1", "Precision of '1'",
              "Escalations flagged '1' (recall)", "Share answered 0", "Accuracy of '0' (NPV)", "Refused: unlike training data",
              "Test tickets with near-copy in training", "Chosen C"]
s2_metrics = ["Pair accuracy (always answering)", "Category accuracy (always answering)", "Pair macro-F1 (always answering)",
              "Share answered with pair", "Pair accuracy when pair given", "Share answered category only",
              "Category accuracy when category only", "Share abstained", "Refused: unlike training data",
              "Test tickets with pair never seen in training", "Unseen-pair tickets given a pair (wrong)",
              "Unseen-pair tickets handled safely", "Test tickets with near-copy in training", "Chosen C"]
nxt = summary_block(ws, 4, "Stage 1: escalation", ws1, s1, s1_metrics)
summary_block(ws, nxt, "Stage 2: category / sub-category pair", ws2, s2, s2_metrics)
ws.column_dimensions["A"].width = 44
for c in range(2, 14):
    ws.column_dimensions[get_column_letter(c)].width = 15
ws.freeze_panes = "B6"

# ---------------- ablation ----------------
abl_order = ["features", "fold", "AUC", "PR_AUC", "Brier", "coverage", "pred_1_share", "precision_1", "recall_1", "NPV_0", "C"]
wsa, abl = sheet_from_df("S1 ablation", T["stage1_ablation_folds"][abl_order],
                         "Stage 1 inputs comparison (template grouping), per outer fold",
                         "structured = priority + support tier only · embedding = description text only · both = final model. Summary below the fold rows.",
                         S1_LABELS)
summary_block(wsa, 4 + len(abl) + 2, "Summary across folds", wsa, abl,
              ["ROC-AUC", "PR-AUC", "Brier score", "Share answered 0 or 1", "Share answered 1", "Precision of '1'",
               "Escalations flagged '1' (recall)", "Accuracy of '0' (NPV)"],
              key_col="Inputs", keys=list(dict.fromkeys(abl["Inputs"])), last_row=4 + len(abl))


# ---------------- tables with totals ----------------
def with_totals(name, df, title, note, label_col):
    ws, d = sheet_from_df(name, df, title, note)
    first, last = 5, 4 + len(d)
    num_cols = [j for j, c in enumerate(d.columns, 1) if c != label_col and pd.api.types.is_numeric_dtype(d[c])]
    tr = last + 1
    ws.cell(row=tr, column=1, value="Total").font = BOLD
    for j in num_cols:
        L = get_column_letter(j)
        c = ws.cell(row=tr, column=j, value=f"=SUM({L}{first}:{L}{last})"); c.font = BOLD; c.number_format = "#,##0"
    return ws, d, first, last, tr, num_cols


for name, key, title, note in [
    ("S1 decisions", "stage1_decisions_vs_truth", "Stage 1 decisions on the outer test folds vs truth (template grouping)",
     "Counts of new tickets. 'Share escalated' = escalated ÷ row total."),
    ("Old S1 decisions", "old_stage1_decisions", "Old file, final model unchanged: stage 1 decisions vs truth",
     "The old file's 85% escalation rate is far outside training; a '0' on an escalated ticket is a wrong answer."),
]:
    ws, d, first, last, tr, num_cols = with_totals(name, T[key], title, note, "decision")
    a, b = get_column_letter(num_cols[0]), get_column_letter(num_cols[-1])
    tot_col, share_col = len(d.columns) + 1, len(d.columns) + 2
    ws.cell(row=4, column=tot_col, value="Row total"); ws.cell(row=4, column=share_col, value="Share escalated")
    style_header(ws, 4, 2, start=tot_col)
    esc = get_column_letter(list(d.columns).index("escalated") + 1)
    for r in range(first, tr + 1):
        c = ws.cell(row=r, column=tot_col, value=f"=SUM({a}{r}:{b}{r})"); c.font = FONT; c.number_format = "#,##0"
        s = ws.cell(row=r, column=share_col, value=f"=IFERROR({esc}{r}/{get_column_letter(tot_col)}{r},\"n/a\")")
        s.font = FONT; s.number_format = "0.0%"
    widths(ws)

ws, d, first, last, tr, num_cols = with_totals("Old S2 outcomes", T["old_stage2_outcomes"],
                                               "Old file, final model unchanged: stage 2 outcomes by case",
                                               "Excludes 10 old tickets with no category label. 'Different label' answers are listed on 'Old S2 mappings' for review; they are not counted as correct.",
                                               "case")
tot_col = len(d.columns) + 1
ws.cell(row=4, column=tot_col, value="Row total"); style_header(ws, 4, 1, start=tot_col)
a, b = get_column_letter(num_cols[0]), get_column_letter(num_cols[-1])
for r in range(first, tr + 1):
    c = ws.cell(row=r, column=tot_col, value=f"=SUM({a}{r}:{b}{r})"); c.font = FONT; c.number_format = "#,##0"
sr = tr + 2
ws.cell(row=sr, column=1, value="Share of row").font = BOLD
for r_off, r in enumerate(range(first, tr + 1)):
    ws.cell(row=sr + 1 + r_off, column=1, value=f"={get_column_letter(1)}{r}").font = FONT
    for j in num_cols:
        L = get_column_letter(j)
        c = ws.cell(row=sr + 1 + r_off, column=j, value=f"=IFERROR({L}{r}/${get_column_letter(tot_col)}{r},\"n/a\")")
        c.font = FONT; c.number_format = "0.0%"
for j in num_cols:
    ws.cell(row=sr, column=j, value=d.columns[j - 1])
style_header(ws, sr, len(d.columns))
widths(ws)

# ---------------- remaining plain tables ----------------
sheet_from_df("S1 risk bands", T["stage1_risk_bands"], "Stage 1 risk bands: escalation rate observed on the outer test folds (template grouping)",
              "Band edges fixed in advance: Low < 5% · Typical 5–15% · Elevated 15–40% · High ≥ 40% predicted probability. 95% interval = Wilson score interval.")
sheet_from_df("S2 confusions", T["stage2_top_confusions"], "Stage 2: most frequent wrong pairs when forced to answer (template grouping)",
              "Out-of-fold predictions; counts of tickets.")
tmp = T["temporal_holdout"].copy()
tmp["metric"] = [(S1_LABELS if s.startswith("stage 1") else S2_LABELS).get(m, m) for m, s in zip(tmp.metric, tmp.stage)]
ws, _ = sheet_from_df("Temporal", tmp[["stage", "metric", "value"]], "Temporal hold-out: train Jan–Apr 2024, test May–Jun 2024",
                      "Thresholds and C tuned inside Jan–Apr only (template-grouped inner folds).")
for r in range(5, 5 + len(tmp)):
    m = ws.cell(row=r, column=2).value
    ws.cell(row=r, column=3).number_format = "0.0%" if m in PCT else "0.000"
sheet_from_df("Old S1 bands", T["old_stage1_bands"], "Old file: risk bands vs the rate each band promised",
              "Ranking partly transfers (higher bands escalate more), but every band is far above its promised rate: a base-rate shift that needs population-level monitoring.")
sheet_from_df("Old S2 mappings", T["old_stage2_mappings_for_review"], "Old file: label mappings the model proposed (for human review)",
              "Old label → new-scheme answer, with ticket counts. Not counted as correct until a person confirms the mapping.")
sheet_from_df("Examples", T["example_tickets"], "Example tickets scored by the final model", "Includes an off-topic ticket to show refusal.")
sheet_from_df("Model settings", T["final_model_settings"], "Final model settings (fitted on all new data)",
              "Thresholds chosen on inner validation folds only. One similarity cut-off is shared by both stages.")
sheet_from_df("Grouping systems", T["grouping_systems"], "Grouping systems used for cross-validation", "Group counts and sizes over the new file.")
sheet_from_df("Run info", T["run_info"], "Run configuration", "")

wb.save(OUT)
print("saved", OUT, "sheets:", wb.sheetnames)
