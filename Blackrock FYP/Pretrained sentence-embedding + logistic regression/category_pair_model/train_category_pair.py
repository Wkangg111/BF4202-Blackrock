"""Grouped-only nested evaluation of frozen MiniLM embeddings + pair-label LR."""
from pathlib import Path
import json, warnings, hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

def pair_label(category, subcategory):
    return json.dumps([category, subcategory], ensure_ascii=False)

def prepare_data(root=ROOT):
    features = pd.read_csv(root/'prepared_data/ticket_features.csv', dtype={'ticket_id':str})
    targets = pd.read_csv(root/'prepared_data/ticket_targets.csv', dtype={'ticket_id':str})
    df = features.merge(targets[['ticket_id','target_category','target_subcategory']], on='ticket_id', validate='one_to_one')
    good = df.model_text.fillna('').str.strip().ne('')
    for col in ['target_category','target_subcategory']:
        good &= df[col].fillna('').str.strip().ne('')
    excluded = df.loc[~good].copy()
    df = df.loc[good].copy().reset_index(drop=True)
    for col in ['target_category','target_subcategory']:
        df[col] = df[col].str.strip()
    df['pair_label'] = [pair_label(c,s) for c,s in zip(df.target_category,df.target_subcategory)]
    # Connected components: either identical text or same incident/problem component.
    parent = list(range(len(df)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for col in ['model_text','event_group_id']:
        seen = {}
        for i,value in enumerate(df[col]):
            if pd.isna(value) or not str(value).strip(): continue
            if value in seen: parent[find(i)] = find(seen[value])
            else: seen[value] = i
    df['evaluation_group'] = [f'G{find(i):06d}' for i in range(len(df))]
    return df, excluded

def encode_texts(texts, root=ROOT):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    encoder = SentenceTransformer(str(root/'model_approach3/encoder'), device='cpu')
    encoder.max_seq_length = 256
    unique = list(dict.fromkeys(texts))
    vectors = encoder.encode(unique, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    lookup = dict(zip(unique, vectors))
    return np.stack([lookup[t] for t in texts])

def make_model(c):
    return LogisticRegression(C=c, max_iter=3000, class_weight=None, solver='lbfgs', random_state=42)

def grouped_folds(y, groups, n_splits, seed):
    if len(set(groups)) < n_splits: raise ValueError('Not enough independent groups.')
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(np.zeros(len(y)), y, groups))
    for train,test in folds:
        assert set(groups[train]).isdisjoint(groups[test])
        if len(set(y[train])) < 2: raise ValueError('Training fold has fewer than two pair labels.')
    return folds

def select_c(X, y, groups, seed):
    folds = grouped_folds(y, groups, 3, seed)
    rows=[]
    for c in C_GRID:
        pred = np.empty(len(y), dtype=object)
        for train,val in folds:
            model=make_model(c).fit(X[train], y[train])
            if model.n_iter_.max() >= model.max_iter: raise RuntimeError('LR did not converge.')
            pred[val]=model.predict(X[val])
        rows.append({'C':c,'inner_oof_pair_macro_f1':f1_score(y,pred,average='macro',zero_division=0),
                     'inner_oof_pair_accuracy':accuracy_score(y,pred)})
    table=pd.DataFrame(rows)
    best=table.sort_values(['inner_oof_pair_macro_f1','C'],ascending=[False,True]).iloc[0]
    return float(best.C),table

def metrics(frame):
    result={'tickets':len(frame)}
    for level,truth,pred in [('pair','pair_label','predicted_pair'),('category','target_category','predicted_category'),('subcategory','target_subcategory','predicted_subcategory')]:
        result[level+'_accuracy']=accuracy_score(frame[truth], frame[pred])
        result[level+'_macro_f1']=f1_score(frame[truth], frame[pred], average='macro',zero_division=0)
        result[level+'_weighted_f1']=f1_score(frame[truth], frame[pred], average='weighted',zero_division=0)
    return result

def run(root=ROOT, output=None):
    out=Path(output) if output else root/'category_pair_model/results'
    out.mkdir(parents=True,exist_ok=True)
    df,excluded=prepare_data(root)
    df.to_csv(out/'prepared_pairs.csv',index=False);excluded.to_csv(out/'excluded_rows.csv',index=False)
    support=df.groupby(['target_category','target_subcategory','pair_label']).agg(tickets=('ticket_id','size'),groups=('evaluation_group','nunique')).reset_index()
    support.to_csv(out/'pair_support.csv',index=False)
    print(f'{len(df)} tickets; {df.pair_label.nunique()} pairs; {df.evaluation_group.nunique()} connected groups',flush=True)
    X=encode_texts(df.model_text.tolist(),root)
    y=df.pair_label.to_numpy(); groups=df.evaluation_group.to_numpy()
    folds=grouped_folds(y,groups,5,42)
    oof=df.copy();oof['predicted_pair']='';oof['fold']=0;oof['pair_confidence']=np.nan
    oof['pair_seen_in_training']=False
    selection=[];fold_metrics=[];assignments=[]
    for fold,(train,test) in enumerate(folds,1):
        c,table=select_c(X[train],y[train],groups[train],42+fold)
        table['outer_fold']=fold;selection.append(table)
        model=make_model(c).fit(X[train],y[train])
        prob=model.predict_proba(X[test]);pred=model.classes_[prob.argmax(axis=1)]
        oof.loc[test,'predicted_pair']=pred;oof.loc[test,'pair_confidence']=prob.max(axis=1)
        oof.loc[test,'fold']=fold;oof.loc[test,'pair_seen_in_training']=np.isin(y[test],model.classes_)
        assignments.extend({'fold':fold,'ticket_id':df.iloc[i].ticket_id,'role':role,'evaluation_group':groups[i]} for role,idx in [('train',train),('test',test)] for i in idx)
        fold_metrics.append({'fold':fold,'selected_C':c,'train_tickets':len(train),'test_tickets':len(test),'pair_accuracy':accuracy_score(y[test],pred),'pair_macro_f1':f1_score(y[test],pred,average='macro',zero_division=0),'unseen_pair_tickets':int((~np.isin(y[test],model.classes_)).sum())})
        print(f'Fold {fold}: C={c}, pair accuracy={fold_metrics[-1]["pair_accuracy"]:.4f}',flush=True)
    assert oof.fold.gt(0).all() and oof.ticket_id.is_unique
    decoded=oof.predicted_pair.map(json.loads)
    oof['predicted_category']=decoded.map(lambda p:p[0]);oof['predicted_subcategory']=decoded.map(lambda p:p[1])
    oof['pair_correct']=oof.pair_label.eq(oof.predicted_pair)
    oof.to_csv(out/'grouped_oof_predictions.csv',index=False)
    oof.loc[~oof.pair_correct].to_csv(out/'misclassified_pairs.csv',index=False)
    pd.DataFrame(assignments).to_csv(out/'outer_fold_assignments.csv',index=False)
    pd.concat(selection).to_csv(out/'inner_model_selection.csv',index=False)
    pd.DataFrame(fold_metrics).to_csv(out/'grouped_fold_metrics.csv',index=False)
    summary=metrics(oof)
    majority=oof.copy()
    for fold,(train,test) in enumerate(folds,1): majority.loc[test,'predicted_pair']=pd.Series(y[train]).mode().iloc[0]
    decoded=majority.predicted_pair.map(json.loads)
    majority['predicted_category']=decoded.map(lambda p:p[0]);majority['predicted_subcategory']=decoded.map(lambda p:p[1])
    pd.DataFrame([{'model':'embedding_lr_pair',**summary},{'model':'training_majority_pair',**metrics(majority)}]).to_csv(out/'grouped_metrics.csv',index=False)
    for level,truth,pred in [('pair','pair_label','predicted_pair'),('category','target_category','predicted_category'),('subcategory','target_subcategory','predicted_subcategory')]:
        labels=sorted(set(oof[truth])|set(oof[pred]))
        pd.DataFrame(confusion_matrix(oof[truth],oof[pred],labels=labels),index=labels,columns=labels).to_csv(out/f'{level}_confusion_matrix.csv')
        (out/f'{level}_classification_report.json').write_text(json.dumps(classification_report(oof[truth],oof[pred],output_dict=True,zero_division=0),indent=2))
    # Deployment model: choose C with grouped CV on all eligible rows AFTER OOF evaluation.
    c,selection=select_c(X,y,groups,142)
    selection.to_csv(out/'final_model_selection.csv',index=False)
    model=make_model(c).fit(X,y)
    joblib.dump({'classifier':model,'target':'category_subcategory_pair','encoder':'sentence-transformers/all-MiniLM-L6-v2','max_seq_length':256,'normalize_embeddings':True,'selected_C':c},out/'category_pair_model.joblib')
    restored=joblib.load(out/'category_pair_model.joblib')
    assert np.allclose(model.predict_proba(X[:10]),restored['classifier'].predict_proba(X[:10]))
    manifest={'method':'5 outer / 3 inner StratifiedGroupKFold','seed':42,'selection_metric':'pooled inner OOF pair macro F1','input':'prepared model_text only','encoder':'all-MiniLM-L6-v2, frozen, normalized, max length 256','eligible_tickets':len(df),'excluded_tickets':len(excluded),'pairs':int(df.pair_label.nunique()),'groups':int(df.evaluation_group.nunique()),'final_C':c,'pair_target_hash':hashlib.sha256('\n'.join(y).encode()).hexdigest(),'limitations':['Synthetic data: not evidence of real-world accuracy.','Reviewed retrospective labels and assumed initial text.','Unseen pairs cannot be predicted; held-out unseen pairs remain in evaluation.','Grouped evaluation does not measure future temporal drift.','Confidence is uncalibrated; no escalation alert threshold is used.']}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.read_csv(out/'grouped_metrics.csv').round(4).to_string(index=False),flush=True)
    return summary

if __name__=='__main__': run()
