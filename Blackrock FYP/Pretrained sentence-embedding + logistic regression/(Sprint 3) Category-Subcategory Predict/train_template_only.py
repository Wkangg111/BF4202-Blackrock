"""Third experiment: exact text + cosine links ONLY, no event constraints."""
import argparse
import pandas as pd
from train_category_pair import run,HERE

def main(threshold=0.90):
    out=HERE/'results_template_only'
    run(output=out,grouping_threshold=threshold,preserve_event_links=False)
    rows=[]
    for experiment,folder in [('original_grouping',HERE/'results'),('combined_similarity_event',HERE/'results_template_family'),('template_only',out)]:
        table=pd.read_csv(folder/'grouped_metrics.csv')
        table.insert(0,'experiment',experiment);rows.append(table)
    comparison=pd.concat(rows,ignore_index=True)
    comparison.to_csv(out/'comparison_all_groupings.csv',index=False)
    print(comparison.round(4).to_string(index=False))
    d=pd.read_csv(out/'grouped_oof_predictions.csv')
    event_overlap=d.groupby('event_group_id').agg(tickets=('ticket_id','size'),outer_test_folds=('fold','nunique'))
    event_overlap.to_csv(out/'event_cross_fold_audit.csv')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--threshold',type=float,default=0.90);a=p.parse_args();main(a.threshold)
