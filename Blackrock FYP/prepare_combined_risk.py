"""Add escalation OR incident-involvement labels. Run directly in VS Code.
Keeps original escalation outputs unchanged; creates splits/risk and risk_targets.csv.
Incident links are retrospective labels, never input features. They do not imply future onset.
"""
from pathlib import Path
import json
import pandas as pd
import prepare_data as base
ROOT=Path(__file__).resolve().parent

def combined_labels(targets,links,asof):
    t=targets.copy()
    for c in ['created_at','closed_at','label_known_at']:
        t[c]=pd.to_datetime(t[c],errors='coerce')
    links=links.copy();links['start_at']=pd.to_datetime(links.start_at,errors='coerce')
    linked=set(links.ticket_id)
    eligible=set(links.loc[links.start_at.notna() & links.start_at.le(asof),'ticket_id'])
    t['has_escalation']=t.escalation_label.eq(1)
    t['has_incident_link']=t.ticket_id.isin(linked)
    t['incident_observed_by_cutoff']=t.ticket_id.isin(eligible)
    t['risk_label']=pd.Series(pd.NA,index=t.index,dtype='Int64')
    positive=t.has_escalation | t.incident_observed_by_cutoff
    negative=t.escalation_label.eq(0) & ~t.has_incident_link
    t.loc[positive,'risk_label']=1;t.loc[negative,'risk_label']=0
    t['risk_label_basis']='outcome_unknown'
    t.loc[negative,'risk_label_basis']='closed_without_escalation_or_incident_link'
    t.loc[t.has_escalation & ~t.incident_observed_by_cutoff,'risk_label_basis']='escalation_only'
    t.loc[~t.has_escalation & t.incident_observed_by_cutoff,'risk_label_basis']='incident_only'
    t.loc[t.has_escalation & t.incident_observed_by_cutoff,'risk_label_basis']='escalation_and_incident'
    # Association timestamp is unavailable. DO NOT substitute incident start time.
    t['risk_label_known_at']=t.label_known_at
    t.loc[t.incident_observed_by_cutoff & ~t.has_escalation,'risk_label_known_at']=pd.NaT
    t['incident_link_known_at']=pd.NaT
    t['label_timing_verified']=t.risk_label_known_at.notna()
    return t

def run(args):
    report=base.run(args)
    out=args.output.resolve()
    targets=pd.read_csv(out/'ticket_targets.csv')
    links=pd.read_csv(out/'incident_case_links.csv')
    risk=combined_labels(targets,links,pd.Timestamp(report['dates']['as_of']))
    base.save_csv(risk,out/'risk_targets.csv')
    features=pd.read_csv(out/'ticket_features.csv')
    splits=pd.read_csv(out/'split_assignments.csv')
    d=features.merge(risk[['ticket_id','risk_label','risk_label_basis']],on='ticket_id',validate='one_to_one').merge(
        splits[['ticket_id','partition','cross_boundary_event_group']],on='ticket_id',validate='one_to_one')
    eligible=d.risk_label.notna() & d.text_usable & ~d.cross_boundary_event_group & pd.to_datetime(d.created_at).le(pd.Timestamp(report['dates']['as_of']))
    counts={};seen=set()
    cols=['ticket_id','created_at','event_group_id','text_group_id','model_text','risk_label']
    for part in ['train','validation','test']:
        sub=d.loc[eligible & d.partition.eq(part),cols].sort_values(['created_at','ticket_id'])
        base.save_csv(sub,out/'splits/risk'/f'{part}.csv')
        if part!='train':base.save_csv(sub.loc[~sub.text_group_id.isin(seen)],out/'splits/risk'/f'{part}_unseen_text.csv')
        seen.update(sub.text_group_id)
        counts[part]={'tickets':len(sub),'positive':int(sub.risk_label.sum()),'negative':int(sub.risk_label.eq(0).sum())}
    change=risk[['ticket_id','escalation_label','has_incident_link','risk_label','risk_label_basis']]
    base.save_csv(change,out/'audit/combined_label_changes.csv')
    base.save_csv(d.loc[~eligible,['ticket_id','risk_label','text_usable','cross_boundary_event_group']],out/'audit/risk_exclusions.csv')
    summary={'target':'escalation OR incident involvement','labels':{'positive':int(risk.risk_label.eq(1).sum()),
        'negative':int(risk.risk_label.eq(0).sum()),'unknown':int(risk.risk_label.isna().sum())},'splits':counts,
        'positive_reasons':risk.loc[risk.risk_label.eq(1),'risk_label_basis'].value_counts().to_dict(),
        'features':['model_text'],'incident_text_used_as_feature':False,'incident_link_timestamp_available':False,
        'timing':'Retrospective associations present in export; no strict combined historical splits created.',
        'limitations':['Synthetic data','Initial text assumed, not verified','Absence of a link treated as no incident only for closed tickets; export completeness assumed',
        'Incident associations are not necessarily future incidents','Old escalation-only results do not evaluate this target']}
    (out/'risk_preparation_report.json').write_text(json.dumps(summary,indent=2))
    print('\nCOMBINED TARGET\n'+json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    parser=base.parser()
    args=parser.parse_args()
    run(args)
