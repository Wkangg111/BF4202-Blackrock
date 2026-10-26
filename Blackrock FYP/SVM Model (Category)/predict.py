"""Suggest a category and subcategory for new tickets with the saved model.

Usage (from this folder, after running the notebook):
    python predict.py --short "Holdings differ from our records" --description "Holdings for Portfolio PT-123 ..."
    python predict.py --csv new_tickets.csv            # columns: SHORT_DESCRIPTION, CASE_DESCRIPTION
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from text_cleaning import model_text

BASE = Path(__file__).resolve().parent
bundle = joblib.load(BASE / 'results' / 'category_model_svm.joblib')


def encode(texts):
    if bundle['encoder'] == 'tfidf':
        return bundle['vectorizer'].transform(texts)
    from sentence_transformers import SentenceTransformer
    enc_dir = BASE / 'encoder'
    enc = SentenceTransformer(str(enc_dir) if enc_dir.exists() else bundle['encoder_name'], device='cpu')
    return enc.encode(list(texts), normalize_embeddings=True)


def suggest(short_descriptions, descriptions, k=3):
    texts = [model_text(s, d) for s, d in zip(short_descriptions, descriptions)]
    clf = bundle['classifier']
    proba = clf.predict_proba(encode(texts))
    order = np.argsort(-proba, 1)[:, :k]
    out = []
    for i, row in enumerate(order):
        sugg = [(clf.classes_[j], bundle['sub_to_category'][clf.classes_[j]], float(proba[i, j])) for j in row]
        out.append({'subcategory': sugg[0][0], 'category': sugg[0][1], 'probability': round(sugg[0][2], 4),
                    'auto_route': sugg[0][2] >= bundle['threshold'],
                    'top_k': [f'{c} / {s} ({p:.2f})' for s, c, p in sugg]})
    return pd.DataFrame(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--short', default='')
    ap.add_argument('--description', default='')
    ap.add_argument('--csv')
    a = ap.parse_args()
    if a.csv:
        df = pd.read_csv(a.csv)
        res = suggest(df['SHORT_DESCRIPTION'], df['CASE_DESCRIPTION'])
        out = Path(a.csv).with_name(Path(a.csv).stem + '_suggestions.csv')
        pd.concat([df, res], axis=1).to_csv(out, index=False)
        print(f'wrote {out}')
    else:
        pd.set_option('display.max_colwidth', 200)
        print(suggest([a.short], [a.description]).T.to_string(header=False))
