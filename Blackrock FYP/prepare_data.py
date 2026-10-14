"""Prepare the final synthetic Excel exports. Run directly in VS Code or with Python.
No fitted text vectorizer or model is created here: fit those on training rows only.
"""
from __future__ import annotations
import argparse, hashlib, html, json, re, unicodedata
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
FILES={'tickets':('Case_details_synthetic_v3.1.xlsx','Sheet1'),
       'escalations':('Escalation_data_synthetic.xlsx','Escalation data'),
       'incidents':('Incidents_synthetic.xlsx','Export')}
MAPS={
 'tickets':{'Number':'ticket_id','Abbreviation':'client_id','Created':'created_at','Category':'category','Sub Category':'subcategory','Client Support':'client_support','Peer Group':'peer_group','Assignment group':'assignment_group','Short description':'short_description','Priority':'priority','State':'state','Updated':'updated_at','Value':'value','Escalation':'reported_escalation','First Response Time':'first_response_at','Closed':'closed_at','Resolved':'resolved_at','Escalation Number':'latest_escalation_id','Escalation Type':'latest_escalation_type','Escalation Count':'reported_escalation_count','Related Incident':'incident_id','masked_description':'description'},
 'escalations':{'ESC_NUMBER':'escalation_id','SOURCE_RECORD':'ticket_id','CLIENT':'client_id','ESC_STATE':'state','APPROVAL_STATUS':'approval_status','ESCALATION_TYPE':'escalation_type','ASSIGNMENT_GROUP':'assignment_group','MANAGER_ESCALATION_GROUP':'manager_group','PRODUCT_EXPERT_ESCALATION_GROUP':'specialist_group','REASON':'reason','ESC_JUSTIFICATION':'justification','SYS_CREATED_ON':'record_created_at','OPENED_AT':'opened_at','UPDATED_ON':'updated_at','CLOSED_AT':'closed_at','MADE_SLA':'made_sla'},
 'incidents':{'ID':'incident_id','Incident Summary':'summary','Description':'description','Problem Number':'problem_id','Client Impact':'client_impact','Client BID (mins)':'business_impact_minutes','Client':'client_id','Start Time':'start_at','Impacted Configuration Item':'configuration_item','Root Cause 1':'root_cause_1','Root Cause 2':'root_cause_2','Root Cause 3':'root_cause_3','Related Case Numbers':'related_case_numbers','Affected Client Codes':'affected_client_codes','Related Case Count':'reported_case_count'}}
KEYS={'tickets':'ticket_id','escalations':'escalation_id','incidents':'incident_id'}
DATES={'tickets':['created_at','updated_at','first_response_at','resolved_at','closed_at'],
       'escalations':['record_created_at','opened_at','updated_at','closed_at'], 'incidents':['start_at']}
JP=re.compile(r'[\u3040-\u30ff\u3400-\u9fff]')
NOTE=re.compile(r'\[\s*(?:support note|post-receipt linkage note)[^\]]*\]',re.I)
IDS=re.compile(r'\b(?:TKT|ESX|EVT|PRM)\d+\b',re.I)

def norm(value):
    if pd.isna(value):return ''
    value=unicodedata.normalize('NFKC',html.unescape(str(value)))
    value=value.replace('\\r\\n','\n').replace('\\n','\n').replace('\r\n','\n').replace('\r','\n')
    return value.strip()

def translate_known(text):
    """Translate only the three observed Japanese templates; do not guess new text."""
    text=text.replace('お世話になっております。','Hello Support,').replace('よろしくお願いいたします。','Thank you for your assistance.')
    text=re.sub(r'(Portfolio PF-\d+)の保有明細が当社記録と一致しません。ご確認をお願いいたします。',r'The holdings for \1 do not match our records. Please investigate.',text)
    text=re.sub(r'(Portfolio PF-\d+)の新規設定をお願いいたします。',r'Please set up the new \1.',text)
    text=re.sub(r'(Instrument IN-\d+)の属性が正しくないようです。ご確認ください。',r'The attributes for \1 appear incorrect. Please verify.',text)
    return text

def clean_text(value, title=False):
    text=translate_known(norm(value))
    text=NOTE.sub('',text)
    text=re.sub(r'<[^>]+>',' ',text)
    if title:text=re.sub(r'^(?:(?:RE|FW|FWD)\s*:\s*)+','',text,flags=re.I)
    else:
        # Keep the first request body before the signature; do not append later threads.
        text=re.split(r'(?im)^\s*(?:regards|kind regards|best regards|many thanks|thanks|thank you for your assistance)[,!\.]*\s*$',text,maxsplit=1)[0]
        text=re.sub(r'(?im)^\s*(?:hello support|hi team|dear support|hello team|hi support)[,!\.]*\s*$','',text)
    text=IDS.sub(' ',text)
    text=re.sub(r'\bCL\d{4}\b','CLIENT',text,flags=re.I)
    text=re.sub(r'\bPF-\d+\b','PORTFOLIO',text,flags=re.I)
    text=re.sub(r'\bIN-\d+\b','INSTRUMENT',text,flags=re.I)
    text=re.sub(r'\bF-\d+\b','FUND',text,flags=re.I)
    text=re.sub(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b','EMAIL',text)
    return re.sub(r'\s+',' ',text).strip()

def symptom_text(summary,description):
    text=norm(description)
    m=re.search(r'(?is)what happened\s*:\s*(.*?)(?=\n\s*(?:who was affected|how it was found|business impact|how it was fixed|follow-up)\s*:|$)',text)
    # Unstructured postmortems are not safe features: fall back to the summary.
    body=m.group(1).strip() if m else ''
    a,b=clean_text(summary,True),clean_text(body)
    return ' '.join(dict.fromkeys(s for s in [a,b] if s)),not bool(m)

def split_list(v):return sorted(set(x.strip().upper() for x in norm(v).split(',') if x.strip()))

def event_labels(t,e,asof):
    """Any raised event counts; manager approval is intentionally never inspected."""
    e=e.copy()
    e['event_available_at']=e[['opened_at','record_created_at']].max(axis=1)
    observed=e.loc[e.event_available_at.le(asof)]
    g=observed.groupby('ticket_id').agg(escalation_count=('escalation_id','size'),first_escalation_at=('opened_at','min'),first_escalation_known_at=('event_available_at','min'))
    out=t[['ticket_id','created_at','closed_at']].merge(g,on='ticket_id',how='left',validate='one_to_one')
    out['escalation_count']=out.escalation_count.fillna(0).astype(int)
    out['escalation_label']=pd.Series(pd.NA,index=out.index,dtype='Int64')
    positive=out.escalation_count.gt(0)
    # If a supplied future escalation is already known in the export, do not call that case negative at an earlier as-of.
    future=set(e.loc[e.event_available_at.gt(asof),'ticket_id'])
    negative=~positive & out.closed_at.notna() & out.closed_at.le(asof) & ~out.ticket_id.isin(future)
    out.loc[positive,'escalation_label']=1;out.loc[negative,'escalation_label']=0
    out['label_known_at']=pd.NaT
    out.loc[positive,'label_known_at']=out.loc[positive,'first_escalation_known_at']
    out.loc[negative,'label_known_at']=out.loc[negative,'closed_at']
    out['label_basis']='outcome_unknown'
    out.loc[positive,'label_basis']='any_escalation_raised'
    out.loc[negative,'label_basis']='closed_without_escalation'
    return out

def partition(created,train_end,validation_end):
    return pd.Series(['train' if x<train_end else 'validation' if x<validation_end else 'test' for x in created],index=created.index)

def save_csv(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,encoding='utf-8-sig',date_format='%Y-%m-%d %H:%M:%S')

def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    issues=[];changes=[];manifest={};frames={};dedups={}
    def issue(source,key,code,detail,severity='warning'):
        issues.append({'source':source,'record_id':key,'severity':severity,'issue':code,'detail':str(detail)})
    for kind,(fname,sheet) in FILES.items():
        path=args.input_dir/fname
        raw=pd.read_excel(path,sheet_name=sheet,dtype=object)
        missing=set(MAPS[kind])-set(raw.columns)
        if missing:raise ValueError(f'{fname}: missing columns {sorted(missing)}')
        manifest[kind]={'file':fname,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sheet':sheet,'source_rows':len(raw)}
        d=raw[list(MAPS[kind])].rename(columns=MAPS[kind]).copy()
        for c in d:
            if c not in DATES[kind]:d[c]=d[c].map(norm)
        for c in ['ticket_id','incident_id','escalation_id','latest_escalation_id','problem_id','client_id']:
            if c in d:d[c]=d[c].str.upper()
        key=KEYS[kind]
        if d[key].eq('').any():raise ValueError(f'{kind}: missing primary key')
        dedups[kind]=int(d.duplicated().sum());d=d.drop_duplicates().reset_index(drop=True)
        if d[key].duplicated().any():raise ValueError(f'{kind}: conflicting duplicate primary keys')
        for c in DATES[kind]:
            before=d[c].map(norm)
            d[c]=pd.to_datetime(d[c],errors='coerce',format='mixed')
            for rid in d.loc[before.ne('') & d[c].isna(),key]:issue(kind,rid,'invalid_datetime',c,'error')
        if kind=='tickets' and d.created_at.isna().any():raise ValueError('Ticket Created is missing or invalid.')
        if kind=='escalations' and d[['opened_at','record_created_at']].isna().any().any():raise ValueError('Escalation opening/record creation timestamp missing or invalid.')
        frames[kind]=d
    t,e,i=frames['tickets'],frames['escalations'],frames['incidents']
    for d,cols in [(t,['reported_escalation_count']),(i,['business_impact_minutes','reported_case_count'])]:
        for c in cols:
            d[c]=pd.to_numeric(d[c],errors='coerce')
            if d[c].isna().any() or d[c].lt(0).any():raise ValueError(f'Invalid nonnegative numeric field: {c}')
    e['approval_status']=e.approval_status.str.lower()
    e['made_sla']=e.made_sla.str.lower().map({'true':True,'false':False,'1':True,'0':False})
    if e.made_sla.isna().any():raise ValueError('Unrecognised MADE_SLA boolean.')
    if (~e.ticket_id.isin(t.ticket_id)).any():raise ValueError('Escalation references an absent ticket.')
    if (~t.loc[t.incident_id.ne(''),'incident_id'].isin(i.incident_id)).any():raise ValueError('Ticket references an absent incident.')
    lookup=t.set_index('ticket_id')
    e['source_row_index']=range(len(e))
    for rid,row in e.iterrows():
        ticket=lookup.loc[row.ticket_id]
        if row.client_id!=ticket.client_id:raise ValueError(f'Client mismatch for {row.escalation_id}')
        if not row.assignment_group:
            e.at[rid,'assignment_group']=ticket.assignment_group
            changes.append({'source':'escalations','record_id':row.escalation_id,'field':'assignment_group','action':'filled_from_linked_ticket','detail':ticket.assignment_group})
        if row.opened_at<ticket.created_at:raise ValueError(f'Escalation predates ticket: {row.escalation_id}')
        if pd.notna(ticket.closed_at) and row.opened_at>ticket.closed_at:issue('escalations',row.escalation_id,'raised_after_ticket_closure',row.ticket_id,'error')
        elif pd.notna(ticket.resolved_at) and row.opened_at>ticket.resolved_at:issue('escalations',row.escalation_id,'raised_after_resolution_before_closure','Retained: all raised escalations count.')
        if pd.notna(row.closed_at) and row.closed_at<row.opened_at:issue('escalations',row.escalation_id,'closed_before_opened','Timestamp inconsistency','error')
    counts=e.ticket_id.value_counts()
    latest=e.sort_values(['opened_at','escalation_id']).groupby('ticket_id').tail(1).set_index('ticket_id')
    for _,r in t.iterrows():
        n=int(counts.get(r.ticket_id,0))
        if n!=r.reported_escalation_count:issue('tickets',r.ticket_id,'escalation_count_mismatch',f'source={r.reported_escalation_count}; events={n}')
        if (r.reported_escalation.casefold()=='escalated')!=(n>0):issue('tickets',r.ticket_id,'escalation_flag_mismatch','Labels use event rows.')
        if r.latest_escalation_id and (r.ticket_id not in latest.index or r.latest_escalation_id!=latest.loc[r.ticket_id,'escalation_id']):issue('tickets',r.ticket_id,'latest_escalation_mismatch',r.latest_escalation_id)
        for a,b in [('created_at','resolved_at'),('resolved_at','closed_at'),('created_at','first_response_at'),('created_at','updated_at')]:
            if pd.notna(r[a]) and pd.notna(r[b]) and r[b]<r[a]:issue('tickets',r.ticket_id,'timestamp_order',f'{b} before {a}','error')
    # A union of both directions retains every explicit link and exposes disagreements.
    case_links={(r.ticket_id,r.incident_id) for r in t.itertuples() if r.incident_id}
    incident_links=set()
    for r in i.itertuples():
        ids=split_list(r.related_case_numbers)
        if len(ids)!=r.reported_case_count:issue('incidents',r.incident_id,'case_count_mismatch',f'{r.reported_case_count} vs {len(ids)}')
        for tid in ids:
            if tid not in lookup.index:raise ValueError(f'Incident references absent ticket {tid}')
            incident_links.add((tid,r.incident_id))
            if r.client_id!='MULTIPLE CLIENTS' and r.client_id!=lookup.loc[tid,'client_id']:issue('incidents',r.incident_id,'single_client_mismatch',tid,'error')
    for tid,inc in case_links ^ incident_links:issue('links',tid,'one_sided_incident_link',inc)
    links=pd.DataFrame(sorted(case_links|incident_links),columns=['ticket_id','incident_id'])
    links=links.merge(t[['ticket_id','created_at']],on='ticket_id').merge(i[['incident_id','problem_id','start_at']],on='incident_id')
    links['incident_precedes_ticket']=links.start_at.le(links.created_at)
    links['hours_after_incident']=(links.created_at-links.start_at).dt.total_seconds()/3600
    # No reliable initial snapshots exist. Model text is a cleaned candidate, not verified historical reconstruction.
    text_rows=[]
    for r in t.itertuples():
        title,body=clean_text(r.short_description,True),clean_text(r.description)
        combined=' '.join(dict.fromkeys(x for x in [title,body] if x))
        translated=bool(JP.search(r.description))
        if translated:changes.append({'source':'tickets','record_id':r.ticket_id,'field':'description','action':'translated_known_japanese_template','detail':'Fixed, reviewed translation dictionary; source retained.'})
        note=bool(NOTE.search(r.description))
        if note:changes.append({'source':'tickets','record_id':r.ticket_id,'field':'description','action':'removed_post_event_link_note','detail':'Removed from model text only.'})
        if JP.search(combined):issue('tickets',r.ticket_id,'untranslated_text','Excluded from model splits.','error')
        if not combined:issue('tickets',r.ticket_id,'empty_model_text','Excluded from model splits.','error')
        text_rows.append([r.ticket_id,r.created_at,title,body,combined,bool(combined) and not bool(JP.search(combined)),False,not bool(r.short_description),not bool(r.description),note,translated,hashlib.sha256(combined.casefold().encode()).hexdigest()[:20]])
    features=pd.DataFrame(text_rows,columns=['ticket_id','created_at','short_description_clean','description_clean','model_text','text_usable','initial_snapshot_verified','missing_title','missing_description','removed_incident_note','translated_japanese','text_group_id'])
    # Cleaned historical export is English; original workbook remains the source of truth.
    t['description']=t.description.map(translate_known)
    if args.assume_initial_fields:
        for c in ['category','subcategory','priority','peer_group']:features[c]=t[c].replace('', '__MISSING__')
    # The maximum observed timestamp is a reproducible sample cutoff, not proof of real export completeness.
    observed_dates=pd.concat([d[c] for k,d in frames.items() for c in DATES[k]])
    asof=pd.Timestamp(args.as_of) if args.as_of else observed_dates.max()
    sorted_dates=t.created_at.sort_values().reset_index(drop=True)
    tr=pd.Timestamp(args.train_end) if args.train_end else sorted_dates.iloc[int(len(t)*.70)].floor('D')
    va=pd.Timestamp(args.validation_end) if args.validation_end else sorted_dates.iloc[int(len(t)*.85)].floor('D')
    if not t.created_at.min()<tr<va<=asof:raise ValueError('Require first creation < train-end < validation-end <= as-of.')
    labels=event_labels(t,e,asof)
    labels=labels.merge(t[['ticket_id','category','subcategory','priority','assignment_group','updated_at','resolved_at']],on='ticket_id',validate='one_to_one')
    labels=labels.rename(columns={'category':'target_category','subcategory':'target_subcategory','priority':'target_priority','assignment_group':'target_team'})
    labels['reviewed_labels_known_at']=labels.updated_at
    # Final resolving team is not considered established for unresolved tickets.
    labels['team_label_known_at']=labels.updated_at.where(labels.resolved_at.notna())
    # Group connected incidents sharing a problem record; never split a known group across evaluation sets.
    parent={x:x for x in t.ticket_id}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):parent[find(b)]=find(a)
    for _,grp in links.groupby('problem_id'):
        tids=grp.ticket_id.tolist()
        for tid in tids[1:]:union(tids[0],tid)
    split=t[['ticket_id','created_at']].copy()
    split['partition']=partition(split.created_at,tr,va)
    split['event_group_id']=[find(x) for x in split.ticket_id]
    cross=set(split.groupby('event_group_id').partition.nunique().loc[lambda x:x>1].index)
    split['cross_boundary_event_group']=split.event_group_id.isin(cross)
    features=features.merge(split[['ticket_id','event_group_id']],on='ticket_id',validate='one_to_one')
    policy={'default_input_columns':['model_text'],'identifier_columns':['ticket_id','created_at','event_group_id','text_group_id'],
     'initial_text_status':'Assumed request-time candidate after removing known support annotations; not a verified initial snapshot.',
     'optional_structured_fields_enabled':args.assume_initial_fields,
     'optional_escalation_inputs':['category','subcategory','priority','peer_group'] if args.assume_initial_fields else [],
     'never_use_for_receipt_prediction':['reported_escalation','latest_escalation_id','latest_escalation_type','reported_escalation_count','incident_id','closed_at','resolved_at','updated_at','first_response_at','assignment_group','approval_status','root_cause_1','root_cause_2','root_cause_3'],
     'task_target_exclusions':{'category':['category','subcategory','value'],'subcategory':['subcategory','value'],'priority':['priority'],'team':['assignment_group']},
     'fit_after_split':['TF-IDF vocabulary/IDF','encoders','scalers','feature selection','oversampling'],
     'incident_retrieval_rule':'Require available_at < ticket.created_at. Unknown availability is ineligible. Start Time alone is not a publication timestamp.'}
    ref=[]
    for r in i.itertuples():
        text,fallback=symptom_text(r.summary,r.description)
        avail=pd.Timestamp(args.incidents_available_at) if args.incidents_available_at else pd.NaT
        if pd.notna(avail) and pd.notna(r.start_at) and avail<r.start_at:raise ValueError('Incident availability cannot precede its start.')
        ref.append([r.incident_id,r.problem_id,r.start_at,avail,r.configuration_item,text,fallback,bool(text),pd.notna(avail) and bool(text)])
    reference=pd.DataFrame(ref,columns=['incident_id','problem_id','start_at','available_at','configuration_item','symptom_text','summary_fallback','text_usable','availability_confirmed'])
    tasks={'escalation':('escalation_label','label_known_at'),'category':('target_category','reviewed_labels_known_at'),'subcategory':('target_subcategory','reviewed_labels_known_at'),'priority':('target_priority','reviewed_labels_known_at'),'team':('target_team','team_label_known_at')}
    joined=features.merge(labels,on=['ticket_id','created_at'],validate='one_to_one').merge(split[['ticket_id','partition','cross_boundary_event_group']],on='ticket_id')
    exclusion=[];split_stats={};strict_stats={};class_stats=[];repeat_stats={}
    for task,(target,knowncol) in tasks.items():
        ready=joined[target].notna() & joined[target].astype(str).ne('')
        cutoff=joined.partition.map({'train':tr,'validation':va,'test':asof})
        timely=joined[knowncol].notna() & joined[knowncol].le(cutoff)
        before_asof=joined.created_at.le(asof)
        mature=joined[knowncol].notna() & joined[knowncol].le(asof)
        common=ready & before_asof & joined.text_usable & ~joined.cross_boundary_event_group
        eligible=common & mature
        strict_eligible=common & timely
        for row in joined.loc[~eligible].itertuples():
            idx=row.Index;why=[]
            if not ready.loc[idx]:why.append('missing_or_unknown_target')
            if not mature.loc[idx]:why.append('label_not_known_by_as_of')
            if not before_asof.loc[idx]:why.append('ticket_after_as_of')
            if not row.text_usable:why.append('unusable_text')
            if row.cross_boundary_event_group:why.append('incident_problem_group_crosses_time_boundary')
            exclusion.append([row.ticket_id,task,'retrospective',row.partition,'; '.join(why)])
        for row in joined.loc[~strict_eligible].itertuples():
            why=[];idx=row.Index
            if not ready.loc[idx]:why.append('missing_or_unknown_target')
            if not timely.loc[idx]:why.append('label_not_known_by_partition_cutoff')
            if not before_asof.loc[idx]:why.append('ticket_after_as_of')
            if not row.text_usable:why.append('unusable_text')
            if row.cross_boundary_event_group:why.append('incident_problem_group_crosses_time_boundary')
            exclusion.append([row.ticket_id,task,'strict',row.partition,'; '.join(why)])
        cols=['ticket_id','created_at','event_group_id','text_group_id','model_text',target,knowncol]
        if task=='escalation' and args.assume_initial_fields:cols+=['category','subcategory','priority','peer_group']
        split_stats[task]={};strict_stats[task]={};seen=set()
        for name in ['train','validation','test']:
            sub=joined.loc[eligible & joined.partition.eq(name),cols].sort_values(['created_at','ticket_id'])
            save_csv(sub,output/'splits'/task/(name+'.csv'))
            strict=joined.loc[strict_eligible & joined.partition.eq(name),cols].sort_values(['created_at','ticket_id'])
            save_csv(strict,output/'strict_splits'/task/(name+'.csv'))
            strict_stats[task][name]={'rows':len(strict),'classes':{str(k):int(v) for k,v in strict[target].value_counts().items()}}
            split_stats[task][name]={'rows':len(sub),'classes':{str(k):int(v) for k,v in sub[target].value_counts().items()}}
            overlap=sub.text_group_id.isin(seen)
            repeat_stats[f'{task}/{name}']={'exact_text_seen_in_earlier_partition':int(overlap.sum())}
            # A stricter diagnostic subset exposes performance on text unseen in earlier partitions.
            if name!='train':save_csv(sub.loc[~overlap],output/'splits'/task/(name+'_unseen_text.csv'))
            seen.update(sub.text_group_id)
            for label,n in sub[target].value_counts().items():class_stats.append([task,name,label,int(n)])
    for kind,d in frames.items():save_csv(d,output/'cleaned'/(kind+'.csv'))
    save_csv(features,output/'ticket_features.csv');save_csv(labels,output/'ticket_targets.csv')
    save_csv(reference,output/'incident_reference.csv');save_csv(links,output/'incident_case_links.csv')
    save_csv(split,output/'split_assignments.csv')
    save_csv(pd.DataFrame(exclusion,columns=['ticket_id','task','mode','partition','reason']),output/'audit'/'excluded_rows.csv')
    save_csv(pd.DataFrame(issues,columns=['source','record_id','severity','issue','detail']),output/'audit'/'data_quality_issues.csv')
    save_csv(pd.DataFrame(changes,columns=['source','record_id','field','action','detail']),output/'audit'/'cleaning_actions.csv')
    save_csv(pd.DataFrame(class_stats,columns=['task','partition','target','rows']),output/'audit'/'class_distribution.csv')
    profile=[]
    for kind,d in frames.items():
        for c in d:profile.append([kind,c,int((d[c].isna()|d[c].eq('')).sum()),int(d[c].nunique())])
    save_csv(pd.DataFrame(profile,columns=['dataset','column','missing_count','distinct_count']),output/'audit'/'column_profile.csv')
    fingerprint=features.groupby('text_group_id').agg(rows=('ticket_id','size'),example_ticket=('ticket_id','first')).reset_index()
    save_csv(fingerprint.loc[fingerprint.rows.gt(1)],output/'audit'/'repeated_text_groups.csv')
    report={'sources':manifest,'exact_duplicate_rows_removed':dedups,'counts':{'tickets':len(t),'escalation_events':len(e),'incidents':len(i),'escalated_tickets':int(labels.escalation_label.eq(1).sum()),'non_escalated_tickets':int(labels.escalation_label.eq(0).sum()),'unknown_tickets':int(labels.escalation_label.isna().sum()),'incident_case_links':len(links),'translated_ticket_descriptions':int(features.translated_japanese.sum()),'removed_support_notes':int(features.removed_incident_note.sum()),'missing_descriptions':int(features.missing_description.sum()),'missing_titles':int(features.missing_title.sum()),'unusable_model_text':int((~features.text_usable).sum()),'missing_category_targets':int(labels.target_category.eq('').sum()),'cross_boundary_event_group_tickets':int(split.cross_boundary_event_group.sum()),'unique_model_texts':int(features.text_group_id.nunique())},
      'dates':{'as_of':str(asof),'as_of_inferred':not bool(args.as_of),'train_end_exclusive':str(tr),'validation_end_exclusive':str(va),'ticket_created_min':str(t.created_at.min()),'ticket_created_max':str(t.created_at.max())},
      'split_counts':split_stats,'strict_split_counts':strict_stats,'primary_split_mode':'retrospective: creation ordered; all labels observed by as-of, not necessarily by each historical boundary','text_overlap':repeat_stats,'issue_counts':pd.DataFrame(issues).get('issue',pd.Series(dtype=str)).value_counts().to_dict(),
      'limitations':['Synthetic data; performance does not establish production accuracy.','Only one month of ticket arrivals; class balance is not representative by assumption.','Primary splits are RETROSPECTIVE. They use labels known by the final as-of and cannot establish what could have been trained or tuned at the original boundaries. strict_splits enforce that stronger timing rule, but are not viable for all tasks with this short dataset.','No verified initial text/field history. Default models should use model_text only and acknowledge the snapshot assumption.','Category, priority and team are reviewed targets. Their latest Updated time is used conservatively as a label-availability proxy. Unresolved team labels are excluded.','No incident publication timestamps. Default incident reference rows are not eligible for time-safe retrieval until availability is supplied.','Incident links are associations with prior incidents, not future-incident outcomes.','Time split is purged of incident/problem groups spanning boundaries. Repeated synthetic texts remain; unseen_text subsets are supplied separately.','Do not concatenate split files, include targets as features, or fit text transforms on validation/test data.']}
    (output/'feature_policy.json').write_text(json.dumps(policy,indent=2),encoding='utf-8')
    (output/'preparation_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    md=['# Data preparation report','', 'Prepared from the three finalised synthetic workbooks. Original Excel inputs are retained unchanged.','', '## Cleaning and labels','']
    md += [f'- {k.replace("_"," ")}: **{v}**' for k,v in report['counts'].items()]
    md += ['', 'Escalation label 1 means ANY raised escalation, regardless of manager approval. Label 0 requires closure without any event. Open/unresolved outcomes remain unknown. Multiple events are aggregated to one ticket; no ticket rows are multiplied by joins.', '', 'Positive label availability uses the first event observable in the export (the later of OPENED_AT and SYS_CREATED_ON). Negative availability uses Closed. Dates retain the source timezone convention; no timezone was invented.', '', 'Missing targets are not invented. Missing descriptions fall back to titles. Known Japanese templates are translated; original descriptions remain in input/; cleaned historical descriptions are translated to English. Model text removes support notes, record IDs, email signatures and variable client/portfolio/instrument/fund identifiers. Missing escalation assignment groups are filled from the documented linked-ticket field and logged.', '', '## Retrospective prototype splits', '', f'Train creation < {tr}; validation creation < {va}; test thereafter. Observation cutoff: {asof} (inferred from maximum observed timestamp unless overridden). These splits use all outcomes known by the final observation cutoff, including outcomes recorded after the historical training/validation boundaries. They are suitable for a retrospective synthetic prototype, NOT a strict prospective backtest. Incident/problem groups crossing partitions are excluded from all splits.', '', '| Task | Train | Validation | Test |','|---|---:|---:|---:|']
    for task,s in split_stats.items():md.append(f'| {task} | {s["train"]["rows"]} | {s["validation"]["rows"]} | {s["test"]["rows"]} |')
    md += ['', '## Strict timing diagnostic splits', '', 'strict_splits/ requires outcomes to be observable by each historical partition cutoff. This reveals the short follow-up problem; do not fit or select a model on an empty or single-class validation set.', '', '| Task | Train | Validation | Test |','|---|---:|---:|---:|']
    for task,s in strict_stats.items():md.append(f'| {task} | {s["train"]["rows"]} | {s["validation"]["rows"]} | {s["test"]["rows"]} |')
    md += ['', 'Full class counts are in audit/class_distribution.csv. Small or missing classes are a limitation, not a reason to shuffle future data into training. The optional *_unseen_text.csv files exclude texts seen in earlier primary partitions and may be small or empty.', '', '## Data quality findings', '']
    md += [f'- {k}: {v}' for k,v in report['issue_counts'].items()] or ['- No integrity issues detected.']
    md += ['', '## Important limitations','']+[f'- {v}' for v in report['limitations']]
    md += ['', '## Files to use', '', '- Start with splits/escalation/train.csv, validation.csv and test.csv. Use model_text as X and escalation_label as y.', '- Use the corresponding category, subcategory, priority or team folders for those tasks. Identifier/date/group columns are metadata, not model inputs.', '- ticket_features.csv and ticket_targets.csv are master preparation tables, not a combined training set.', '- cleaned/ retains normalized historical fields for traceability, not unrestricted model inputs.', '- incident_reference.csv contains symptom-only text; incident_case_links.csv is an audit/grouping table.', '- audit/excluded_rows.csv explains each task-specific exclusion. Other audit files record missingness, distributions and changes.', '- feature_policy.json defines allowed and forbidden inputs.', '- Fit TF-IDF, scaling, encoders and any resampling only after selecting training rows. No model fitting is performed by this preparation script.','']
    (output/'PREPARATION_REPORT.md').write_text('\n'.join(md),encoding='utf-8')
    print(json.dumps({'output':str(output),'counts':report['counts'],'dates':report['dates'],'splits':{k:{p:v['rows'] for p,v in s.items()} for k,s in split_stats.items()},'issues':report['issue_counts']},indent=2))
    return report

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=ROOT/'input')
    p.add_argument('--output',type=Path,default=ROOT/'prepared_data')
    p.add_argument('--as-of',help='Export observation cutoff; default maximum timestamp in supplied files.')
    p.add_argument('--train-end',help='Exclusive end of training creation dates; default near 70th percentile day.')
    p.add_argument('--validation-end',help='Exclusive end of validation creation dates; default near 85th percentile day.')
    p.add_argument('--incidents-available-at',help='Explicit date at which the whole incident corpus was available. Unknown by default; must not precede an incident start.')
    p.add_argument('--assume-initial-fields',action='store_true',help='Explicitly permit current category/subcategory/priority/peer group as initial escalation features. Default text-only.')
    return p

if __name__=='__main__':
    run(parser().parse_args())
