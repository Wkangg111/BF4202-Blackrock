"""Predict valid pairs from CSV with model_text or short_description + description."""
from pathlib import Path
import argparse,json,joblib
import numpy as np
import pandas as pd
from train_category_pair import HERE, ROOT, encode_texts

def predict(df, model_path=HERE/'results/category_pair_model.joblib'):
    if 'model_text' in df:
        texts=df.model_text.fillna('').astype(str).str.strip()
    else:
        title='short_description' if 'short_description' in df else 'Short description'
        body='description' if 'description' in df else 'Description'
        if title not in df or body not in df: raise ValueError('Supply model_text, or short_description and description.')
        texts=(df[title].fillna('').astype(str)+' '+df[body].fillna('').astype(str)).str.strip()
    if texts.eq('').any(): raise ValueError('Empty ticket text is not allowed.')
    artifact=joblib.load(model_path)
    X=encode_texts(texts.tolist(),ROOT);clf=artifact['classifier'];prob=clf.predict_proba(X)
    pairs=[json.loads(p) for p in clf.classes_[np.argmax(prob,axis=1)]]
    out=df.copy();out['predicted_category']=[p[0] for p in pairs];out['predicted_subcategory']=[p[1] for p in pairs]
    out['pair_confidence']=prob.max(axis=1)
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=HERE/'example_new_tickets.csv');parser.add_argument('--output',type=Path,default=HERE/'new_ticket_predictions.csv');parser.add_argument('--model',type=Path,default=HERE/'results/category_pair_model.joblib');args=parser.parse_args()
    result=predict(pd.read_csv(args.input),args.model);result.to_csv(args.output,index=False);print(result.to_string(index=False))
