"""Prepare train / validation / test partitions for the escalation-risk experiment.

Run from this folder:  python prepare_data.py
Input : source_data/*.xlsx (synthetic case, escalation and incident exports)
Output: data/{train,validation,test}.csv, data/assumed_initial_fields.csv, data/preparation_report.json

Design (mirrors Approach 3):
  * Target  : escalation_label = 1 if the case has any recorded escalation (any approval status).
  * Input   : model_text = cleaned short description + description. Greeting, sign-off, client/instrument/
              fund identifiers and the 'Support note: linked to service issue EVT...' line are removed
              (the support note names an incident, so it would leak incident linkage).
  * Groups  : event_group_id = problem number of the linked incident, otherwise the ticket itself.
  * Split   : chronological 70 / 13 / 17 %. Tickets whose event group straddles a boundary are dropped
              from the later partition so no group appears in two partitions.
"""
import json, re
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
SRC = BASE / 'source_data'
OUT = BASE / 'data'; OUT.mkdir(exist_ok=True)

cases = pd.read_excel(SRC / 'Case_details_synthetic_v3.1.xlsx', sheet_name='Sheet1')
esc = pd.read_excel(SRC / 'Escalation_data_synthetic.xlsx', sheet_name='Escalation data')
inc = pd.read_excel(SRC / 'Incidents_synthetic.xlsx', sheet_name='Export')

# ---- label: any recorded escalation, cross-checked against the escalation export -------------
cases['escalation_label'] = (cases['Escalation'].eq('Escalated')).astype(int)
in_esc_file = cases['Number'].isin(esc['SOURCE_RECORD'])
assert (cases['escalation_label'].astype(bool) == in_esc_file).all(), 'case flag disagrees with escalation file'
assert ((cases['Escalation Count'] > 0) == in_esc_file).all()

# ---- text cleaning ---------------------------------------------------------------------------
GREETING = re.compile(r'^(hello support|dear support|hi team|good morning|hello|hi|お世話になっております)[,。]?\s*$', re.I)
def clean_description(text):
    if pd.isna(text): return ''
    text = re.sub(r'\[Support note:[^\]]*\]', '', str(text))
    lines = [l.strip() for l in text.strip().splitlines()]
    lines = [l for l in lines if l]
    if lines and GREETING.match(lines[0]): lines = lines[1:]
    # sign-off = closing line + "<role>, CL####"
    if lines and re.search(r'CL\d{4}\s*$', lines[-1]):
        lines = lines[:-1]
        if lines and len(lines[-1]) <= 25 and lines[-1].rstrip().endswith((',', '、')): lines = lines[:-1]
    return ' '.join(lines)
def mask_ids(s):
    s = re.sub(r'\bCL\d{4}\b', 'CLIENT', s)
    s = re.sub(r'\bEVT\d+\b', 'INCIDENT', s)
    s = re.sub(r'\bTKT\d+\b', 'TICKET', s)
    s = re.sub(r'\bIN-\d+\b', 'INSTRUMENT', s)
    s = re.sub(r'\bF-\d+\b', 'FUND', s)
    s = re.sub(r'\b(?:Many thanks|Thanks|Kind regards|Best regards|Regards)\s*,?\s*[A-Za-z ]{3,40},\s*CLIENT\b', '', s)  # sign-offs quoted inside forwarded text
    return re.sub(r'\s+', ' ', s).strip()
short = cases['Short description'].fillna('').map(lambda s: mask_ids(str(s)))
body = cases['masked_description'].map(clean_description).map(mask_ids)
cases['model_text'] = (short + '. ' + body).str.strip(' .').where(body.ne('') & short.ne(''), short.where(short.ne(''), body))
cases['model_text'] = cases['model_text'].str.strip()

n_all = len(cases)
cases = cases[cases['model_text'].ne('')].copy()
n_no_text = n_all - len(cases)

# ---- groups ----------------------------------------------------------------------------------
prob = inc.set_index('ID')['Problem Number']
cases['event_group_id'] = cases['Related Incident'].map(prob).fillna(cases['Number'])
cases['text_group_id'] = pd.factorize(cases['model_text'])[0]
cases['text_group_id'] = 'TXT' + cases['text_group_id'].astype(str).str.zfill(5)
cases['created_at'] = pd.to_datetime(cases['Created'])

# ---- chronological split with group separation ----------------------------------------------
cases = cases.sort_values(['created_at', 'Number']).reset_index(drop=True)
n = len(cases)
cut1, cut2 = int(n * 0.70), int(n * 0.83)
cases['split'] = ['train'] * cut1 + ['validation'] * (cut2 - cut1) + ['test'] * (n - cut2)
order = {'train': 0, 'validation': 1, 'test': 2}
first_split = cases.groupby('event_group_id')['split'].transform(lambda s: min(s, key=order.get))
straddle = cases['split'].map(order) > first_split.map(order)      # later member of a group already placed earlier
dropped = cases[straddle]
cases = cases[~straddle].copy()

keep = ['Number', 'created_at', 'event_group_id', 'text_group_id', 'model_text', 'escalation_label']
for s in ['train', 'validation', 'test']:
    d = cases[cases.split == s][keep].rename(columns={'Number': 'ticket_id'})
    d.to_csv(OUT / f'{s}.csv', index=False)

# Assumed initial structured fields (supplied values are treated as the initial ones - unverified).
meta = cases[['Number', 'Category', 'Sub Category', 'Priority']].rename(
    columns={'Number': 'ticket_id', 'Category': 'category', 'Sub Category': 'subcategory', 'Priority': 'priority'})
meta.to_csv(OUT / 'assumed_initial_fields.csv', index=False)

report = {
    'source_cases': n_all, 'dropped_no_text': int(n_no_text),
    'dropped_group_straddles_split': int(len(dropped)),
    'eligible_tickets': int(len(cases)),
    'partitions': {s: {'tickets': int((cases.split == s).sum()),
                       'escalated': int(cases[cases.split == s].escalation_label.sum()),
                       'created_min': str(cases[cases.split == s].created_at.min()),
                       'created_max': str(cases[cases.split == s].created_at.max())} for s in order},
    'unique_model_texts': int(cases.model_text.nunique()),
    'excluded_fields': ['assignment group', 'escalation number/type/count', 'incident/problem linkage text',
                        'client code', 'state', 'resolution timestamps', 'client support tier', 'peer group'],
}
(OUT / 'preparation_report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
