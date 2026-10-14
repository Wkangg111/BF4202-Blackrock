"""Score already-prepared model_text. Run: python predict.py prepared_tickets.csv"""
from pathlib import Path
import argparse
import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('input',type=Path)
 p.add_argument('--output',type=Path,default=Path('new_predictions.csv'))
 a=p.parse_args();base=Path(__file__).resolve().parent
 d=pd.read_csv(a.input)
 if 'model_text' not in d or d.model_text.isna().any() or d.model_text.str.strip().eq('').any():
  raise ValueError('Input must contain nonempty model_text cleaned using the preparation pipeline.')
 if 'ticket_id' in d and not d.ticket_id.is_unique: raise ValueError('Duplicate ticket IDs')
 artifact=joblib.load(base/'results/escalation_model_structured.joblib')
 encoder=SentenceTransformer(str(base/'encoder'),device='cpu')
 encoder.max_seq_length=artifact['max_seq_length']
 X=encoder.encode(d.model_text.tolist(),normalize_embeddings=True,batch_size=32)
 fields=artifact['structured_fields']
 if not set(fields).issubset(d.columns): raise ValueError('Input must also include category, subcategory, priority as initial values.')
 metadata=d[fields].fillna('__MISSING__').replace('', '__MISSING__').astype(str)
 X=hstack([csr_matrix(X),artifact['onehot'].transform(metadata)],format='csr')
 out=d.copy();out['risk_score']=artifact['classifier'].predict_proba(X)[:,1]
 out['alert_threshold']=artifact['threshold'];out['prediction']=(out.risk_score>=artifact['threshold']).astype(int)
 out.to_csv(a.output,index=False);print(f'Saved {a.output}')
if __name__=='__main__': main()
