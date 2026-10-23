"""Second grouped-only experiment, preserving baseline results."""
import argparse
import pandas as pd
from train_category_pair import run,HERE

def main(threshold=0.90):
    out=HERE/'results_template_family'
    run(output=out,grouping_threshold=threshold)
    rows=[]
    for experiment,folder in [('original_grouping',HERE/'results'),('template_family_proxy',out)]:
        table=pd.read_csv(folder/'grouped_metrics.csv')
        table.insert(0,'experiment',experiment);rows.append(table)
    comparison=pd.concat(rows,ignore_index=True)
    comparison.to_csv(out/'comparison_with_original.csv',index=False)
    print(comparison.round(4).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--threshold',type=float,default=0.90);args=p.parse_args();main(args.threshold)
