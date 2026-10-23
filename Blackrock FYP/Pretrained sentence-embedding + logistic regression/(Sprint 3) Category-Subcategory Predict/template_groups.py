"""Label-independent semantic family proxy; connected components preserve ALL links."""
import json, warnings
import numpy as np
import pandas as pd

def build_template_groups(df, embeddings, threshold=0.90, output=None, preserve_event_links=True):
    if not 0 < threshold <= 1: raise ValueError('threshold must be in (0, 1]')
    if len(df)!=len(embeddings): raise ValueError('Embedding rows must match tickets.')
    if not np.isfinite(embeddings).all(): raise ValueError('Non-finite embedding.')
    norms=np.linalg.norm(embeddings,axis=1,keepdims=True)
    if (norms==0).any(): raise ValueError('Zero embedding.')
    X=embeddings/norms
    d=df.copy().reset_index(drop=True);parent=list(range(len(d)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(i)]=find(j)
    seen={}
    for i,g in enumerate(d.evaluation_group if preserve_event_links else d.model_text):
        if g in seen:union(i,seen[g])
        else:seen[g]=i
    # Compare distinct text only; exact copies already share original groups.
    idx=np.flatnonzero(~d.model_text.duplicated().to_numpy())
    similarity=np.clip(X[idx]@X[idx].T,-1,1)
    left,right=np.where(np.triu(similarity>=threshold,k=1))
    edges=[]
    for a,b in zip(left,right):
        i,j=int(idx[a]),int(idx[b]);union(i,j)
        edges.append({'ticket_a':d.iloc[i].ticket_id,'ticket_b':d.iloc[j].ticket_id,'cosine_similarity':float(similarity[a,b]),'text_a':d.iloc[i].model_text,'text_b':d.iloc[j].model_text})
    d['original_evaluation_group']=d.evaluation_group
    d['evaluation_group']=[f'TF{find(i):06d}' for i in range(len(d))]
    if output is not None:
        pd.DataFrame(edges,columns=['ticket_a','ticket_b','cosine_similarity','text_a','text_b']).to_csv(output/'similarity_links.csv',index=False)
        sizes=d.groupby('evaluation_group').agg(tickets=('ticket_id','size'),unique_texts=('model_text','nunique'),original_groups=('original_evaluation_group','nunique'))
        sizes.to_csv(output/'template_group_sizes.csv')
        if sizes.tickets.max()/len(d) > 0.4:
            warnings.warn('A semantic/event component contains over 40% of tickets. Folds are highly imbalanced; interpret results as an exploratory stress test and review grouping links.')
        # Boundary examples are selected by similarity, never by outcome or accuracy.
        a,b=np.triu_indices(len(idx),1);scores=similarity[a,b]
        nearest=np.argsort(np.abs(scores-threshold))[:40]
        pd.DataFrame([{'cosine_similarity':float(scores[k]),'merged_directly':bool(scores[k]>=threshold),'text_a':d.iloc[idx[a[k]]].model_text,'text_b':d.iloc[idx[b[k]]].model_text} for k in nearest]).to_csv(output/'threshold_review_examples.csv',index=False)
        (output/'grouping_diagnostics.json').write_text(json.dumps({'threshold':threshold,'original_groups':int(d.original_evaluation_group.nunique()),'new_groups':int(d.evaluation_group.nunique()),'unique_texts':len(idx),'similarity_edges':len(edges),'largest_group_tickets':int(sizes.tickets.max()),'largest_group_fraction':float(sizes.tickets.max()/len(d)),'preserve_event_links':preserve_event_links,'rule':'cosine >= threshold plus ' + ('original event/text groups' if preserve_event_links else 'exact text only') + '; transitive connected components','status':'Exploratory semantic proxy, not verified source template IDs. Threshold fixed before evaluation, not selected for accuracy.'},indent=2))
    return d
