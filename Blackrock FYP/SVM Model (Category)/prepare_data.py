"""Prepare train / validation / test partitions for the ticket category + subcategory classifier.

Run from this folder:  python prepare_data.py
Input : source_data/Case_details_synthetic_v5.xlsx  (sheet 'Cases')
Output: data/{train,validation,test}.csv     labelled tickets, chronological split
        data/unlabelled.csv                  tickets with no category (routed to Service Desk L1/L2)
        data/label_hierarchy.csv             subcategory -> category map (each subcategory has one parent)
        data/preparation_report.json

Design
  * Target : SUB_CATEGORY (59 classes). CATEGORY (17) is read off the hierarchy, so the two predictions
             can never disagree.
  * Input  : model_text = cleaned short description + cleaned description, i.e. only what the client
             writes when raising the ticket. Assignment group, priority, state, escalation and incident
             fields are excluded (they are set by the desk after triage, or later).
  * Leakage removed from the text:
      - service-form lines 'Request type: <subcategory>' and 'Category: <category>' (the label itself),
        plus the other form header lines (client code, requester role, dates, requested priority);
      - '[Support note: linked to service issue EVR...]' (added by support after triage);
      - the '----- Earlier message -----' quoted copy (duplicate of the body);
      - greeting, sign-off, requester role and client code.
    Identifiers, amounts and dates are masked so the model cannot memorise them; the ID prefix is kept
    as a word (Portfolio / Instrument / Fund ...) because the kind of object is a legitimate clue.
  * Templates: short_template / body_template are the masked texts. They are used only for the
    'unseen wording' stress test in the notebook (hold out whole templates), not as features.
  * Split  : chronological 70 / 13 / 17 % of labelled tickets by CREATED.
"""
import json, re
from pathlib import Path
import pandas as pd
from text_cleaning import clean_short, clean_body

BASE = Path(__file__).resolve().parent
SRC = BASE / 'source_data'
OUT = BASE / 'data'; OUT.mkdir(exist_ok=True)

cases = pd.read_excel(SRC / 'Case_details_synthetic_v5.xlsx', sheet_name='Cases')
n_source = len(cases)

# ---- hierarchy check: every subcategory must have exactly one parent category --------------------
labelled_mask = cases['SUB_CATEGORY'].notna() & cases['CATEGORY'].notna()
parents = cases[labelled_mask].groupby('SUB_CATEGORY')['CATEGORY'].nunique()
assert (parents == 1).all(), f'subcategories with several parents: {list(parents[parents > 1].index)}'
hierarchy = (cases[labelled_mask][['SUB_CATEGORY', 'CATEGORY']].drop_duplicates()
             .sort_values(['CATEGORY', 'SUB_CATEGORY'])
             .rename(columns={'SUB_CATEGORY': 'subcategory', 'CATEGORY': 'category'}))
hierarchy.to_csv(OUT / 'label_hierarchy.csv', index=False)

# ---- text cleaning (shared with the notebook and predict.py) ----------------------------------------

cases['short_text'] = cases['SHORT_DESCRIPTION'].map(clean_short)
cases['body_text'] = cases['CASE_DESCRIPTION'].map(clean_body)
cases['model_text'] = [(f'{s}. {b}' if s and b else (s or b)) for s, b in zip(cases.short_text, cases.body_text)]
cases['is_service_form'] = cases['CASE_DESCRIPTION'].fillna('').str.contains('Request submitted via service form')
cases['is_japanese'] = cases['CASE_DESCRIPTION'].fillna('').str.contains(r'[぀-ヿ]')

# Sanity: the label must not survive in form tickets after cleaning.
form = cases[labelled_mask & cases.is_service_form]
assert not any(f'request type: {sc.lower()}' in t.lower() for sc, t in zip(form.SUB_CATEGORY, form.model_text))

# Templates (for the stress test only).
cases['short_template'] = 'S' + pd.Series(pd.factorize(cases['short_text'].str.lower())[0], index=cases.index).astype(str)
cases['body_template'] = 'B' + pd.Series(pd.factorize(cases['body_text'].str.lower())[0], index=cases.index).astype(str)

no_text = cases['model_text'].eq('')
keep = ['ticket_id', 'created_at', 'model_text', 'short_text', 'body_text', 'short_template', 'body_template',
        'is_service_form', 'is_japanese', 'category', 'subcategory']
cases = cases.rename(columns={'NUMBER': 'ticket_id', 'CREATED': 'created_at',
                              'CATEGORY': 'category', 'SUB_CATEGORY': 'subcategory'})

# ---- unlabelled tickets (the routing use-case) -------------------------------------------------------
unl = cases[~labelled_mask & ~no_text].copy()
unl[keep + ['ASSIGNMENT_GROUP', 'STATE']].to_csv(OUT / 'unlabelled.csv', index=False)

# ---- chronological split of labelled tickets ---------------------------------------------------------
lab = cases[labelled_mask & ~no_text].sort_values(['created_at', 'ticket_id']).reset_index(drop=True)
n = len(lab); cut1, cut2 = int(n * 0.70), int(n * 0.83)
lab['split'] = ['train'] * cut1 + ['validation'] * (cut2 - cut1) + ['test'] * (n - cut2)
for s in ['train', 'validation', 'test']:
    lab[lab.split == s][keep].to_csv(OUT / f'{s}.csv', index=False)

train_short = set(lab[lab.split == 'train'].short_text.str.lower())
report = {
    'source_tickets': n_source,
    'labelled_tickets': int(len(lab)),
    'unlabelled_tickets': int(len(unl)),
    'dropped_labelled_no_text': int((labelled_mask & no_text).sum()),
    'categories': int(lab.category.nunique()),
    'subcategories': int(lab.subcategory.nunique()),
    'service_form_tickets_label_lines_removed': int(lab.is_service_form.sum()),
    'japanese_tickets': int(lab.is_japanese.sum()),
    'short_templates': int(lab.short_text.str.lower().nunique()),
    'body_templates': int(lab.body_text.str.lower().nunique()),
    'partitions': {s: {'tickets': int((lab.split == s).sum()),
                       'subcategories_present': int(lab[lab.split == s].subcategory.nunique()),
                       'created_min': str(lab[lab.split == s].created_at.min()),
                       'created_max': str(lab[lab.split == s].created_at.max()),
                       'short_template_seen_in_train': round(float(
                           lab[lab.split == s].short_text.str.lower().isin(train_short).mean()), 4)}
                   for s in ['train', 'validation', 'test']},
    'excluded_fields': ['ASSIGNMENT_GROUP (set by the desk; it is the routing target)', 'PRIORITY', 'STATE',
                        'ESCALATION*', 'RELATED_INCIDENT and support-note text', 'timestamps after creation',
                        'client code', 'CLIENT_SUPPORT tier'],
    'not_used': {
        'Escalation_data_synthetic_v2.xlsx': 'escalations happen after the ticket is raised and categorised',
        'Interface_Catalog.xls': 'no ticket text mentions an interface name or spec number',
    },
}
(OUT / 'preparation_report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
