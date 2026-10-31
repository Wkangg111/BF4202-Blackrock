"""Ticket text cleaning shared by prepare_data.py, the notebook and predict.py.

Keeps only what the client writes when raising a ticket, removes anything that reveals the label or was
added after triage, and masks identifiers / amounts / dates.
"""
import re

ID_WORD = {'PT': 'portfolio', 'IS': 'instrument', 'FD': 'fund', 'OR': 'order', 'BX': 'benchmark', 'RL': 'rule'}
FORM_HEADER = re.compile(r'^(Request submitted via service form|Request type:|Category:|Client code:|'
                         r'Requested by:|Required by:|Priority requested:)', re.I)
GREETING = re.compile(r'^(good morning|good afternoon|hello support|dear support|hello|hi team|hi|'
                      r'お世話になっております)[,。]?$', re.I)
CLOSING = re.compile(r'^(thanks|many thanks|kind regards|best regards|regards|cheers|'
                     r'よろしくお願いいたします)[,。]?$', re.I)
SIGNATURE = re.compile(r'^[A-Za-z ]{3,40},\s*CX\d{4}$')            # "<role>, CX1234"


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and x != x)


def mask(s: str) -> str:
    """Mask identifiers, amounts and dates; keep the kind of identifier as a word."""
    s = re.sub(r'\bCX\d{4}\b', 'client', s)
    s = re.sub(r'\bEVR\d+\b', 'issue', s)
    s = re.sub(r'\b(PT|IS|FD|OR|BX|RL)-\d+\b', lambda m: ID_WORD[m.group(1)] + ' ref', s)
    s = re.sub(r'\b(USD|EUR|GBP|JPY|AUD|SGD|CHF|CAD|HKD)\s?\d[\d.,]*\s?[mbk]?\b', 'amount', s)
    s = re.sub(r'\b\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}\b', 'date', s)
    s = re.sub(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}\b', 'month', s)
    s = re.sub(r'\d+(?:[.,]\d+)*', 'N', s)
    return re.sub(r'\s+', ' ', s).strip()


def clean_short(text) -> str:
    """Short description without RE:/FW: prefixes, masked."""
    if _missing(text): return ''
    s = re.sub(r'^\s*((RE|FW|FWD)\s*:\s*)+', '', str(text), flags=re.I)
    return mask(s)


def clean_body(text) -> str:
    """Description without the quoted earlier message, support note, service-form header lines
    (which contain the label), greeting and sign-off; masked."""
    if _missing(text): return ''
    s = str(text).split('----- Earlier message -----')[0]
    s = re.sub(r'\[Support note:[^\]]*\]', '', s)
    out = []
    for line in (l.strip() for l in s.splitlines()):
        if not line or FORM_HEADER.match(line) or GREETING.match(line) or CLOSING.match(line) or SIGNATURE.match(line):
            continue
        out.append(re.sub(r'^Details:\s*', '', line))
    return mask(' '.join(out))


def model_text(short_description, description) -> str:
    s, b = clean_short(short_description), clean_body(description)
    return f'{s}. {b}' if s and b else (s or b)
