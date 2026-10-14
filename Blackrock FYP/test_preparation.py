import json,unittest,hashlib
from pathlib import Path
import pandas as pd
from prepare_data import ROOT,FILES,event_labels,clean_text,symptom_text,JP

P=ROOT/'prepared_data'
class PreparationTests(unittest.TestCase):
    def test_every_approval_status_counts(self):
        statuses=['approved','requested','rejected','not required','not requested']
        t=pd.DataFrame({'ticket_id':list('ABCDEFG'),'created_at':pd.to_datetime(['2024-01-01']*7),'closed_at':pd.to_datetime(['2024-01-10']*6+[None])})
        e=pd.DataFrame({'ticket_id':list('ABCDE'),'escalation_id':list('12345'),'opened_at':pd.to_datetime(['2024-01-02']*5),'record_created_at':pd.to_datetime(['2024-01-03']*5),'approval_status':statuses})
        y=event_labels(t,e,pd.Timestamp('2024-02-01')).set_index('ticket_id')
        self.assertTrue(y.loc[list('ABCDE'),'escalation_label'].eq(1).all())
        self.assertEqual(y.loc['F','escalation_label'],0);self.assertTrue(pd.isna(y.loc['G','escalation_label']))
        self.assertTrue(y.loc[list('ABCDE'),'label_known_at'].eq(pd.Timestamp('2024-01-03')).all())

    def test_future_event_not_observable(self):
        t=pd.DataFrame({'ticket_id':['A'],'created_at':pd.to_datetime(['2024-01-01']),'closed_at':pd.to_datetime([None])})
        e=pd.DataFrame({'ticket_id':['A'],'escalation_id':['E'],'opened_at':pd.to_datetime(['2024-01-02']),'record_created_at':pd.to_datetime(['2024-01-04'])})
        y=event_labels(t,e,pd.Timestamp('2024-01-03'))
        self.assertTrue(pd.isna(y.escalation_label.iloc[0]))

    def test_post_event_note_cannot_change_model_text(self):
        base='Hello Support,\nThe chart is empty.'
        self.assertEqual(clean_text(base),clean_text(base+'\n[Support note: linked to service issue EVT0200005]'))
        self.assertEqual(clean_text('[Support note: linked to service issue EVT0200005]'),'')

    def test_known_translation_preserves_issue(self):
        text=clean_text('Portfolio PF-2271の保有明細が当社記録と一致しません。ご確認をお願いいたします。')
        self.assertIn('do not match our records',text);self.assertIsNone(JP.search(text))

    def test_incident_postmortem_excluded(self):
        text,fallback=symptom_text('Processing delayed','What happened: Jobs were delayed.\nWho was affected: Clients.\nHow it was fixed: Restarted the SECRET_FIX server.')
        self.assertNotIn('SECRET_FIX',text);self.assertFalse(fallback)
        text,fallback=symptom_text('Delayed jobs','Unstructured root cause and fix')
        self.assertEqual(text,'Delayed jobs');self.assertTrue(fallback)

    def test_saved_labels_match_all_event_rows(self):
        e=pd.read_csv(P/'cleaned/escalations.csv');y=pd.read_csv(P/'ticket_targets.csv')
        self.assertEqual(set(y.loc[y.escalation_label.eq(1),'ticket_id']),set(e.ticket_id))
        self.assertEqual(len(y),2636)

    def test_split_groups_and_time_order(self):
        report=json.loads((P/'preparation_report.json').read_text())
        for folder in ['splits','strict_splits']:
            for task in ['escalation','category','subcategory','priority','team']:
                seen_ids=set();seen_groups=set()
                for name in ['train','validation','test']:
                    d=pd.read_csv(P/folder/task/(name+'.csv'))
                    self.assertFalse(seen_ids & set(d.ticket_id));self.assertFalse(seen_groups & set(d.event_group_id))
                    seen_ids.update(d.ticket_id);seen_groups.update(d.event_group_id)
                    if d.empty:continue
                    created=pd.to_datetime(d.created_at)
                    tr=pd.Timestamp(report['dates']['train_end_exclusive']);va=pd.Timestamp(report['dates']['validation_end_exclusive'])
                    if name=='train':self.assertTrue(created.lt(tr).all())
                    elif name=='validation':self.assertTrue((created.ge(tr)&created.lt(va)).all())
                    else:self.assertTrue(created.ge(va).all())
                    if folder=='strict_splits':
                        known=d.columns[-1];cut={'train':tr,'validation':va,'test':pd.Timestamp(report['dates']['as_of'])}[name]
                        self.assertTrue(pd.to_datetime(d[known]).le(cut).all())

    def test_model_files_and_unknown_availability(self):
        for name in ['train','validation','test']:
            d=pd.read_csv(P/'splits/escalation'/(name+'.csv'))
            self.assertEqual(set(d.escalation_label),{0,1})
            self.assertFalse(d.model_text.isna().any())
            self.assertFalse(d.model_text.str.contains(r'\b(?:EVT|ESX|TKT|PRM)\d+\b',regex=True).any())
            self.assertNotIn('approval_status',d.columns)
        refs=pd.read_csv(P/'incident_reference.csv')
        self.assertTrue(refs.available_at.isna().all())
        self.assertFalse(refs.availability_confirmed.any())

    def test_source_files_unchanged(self):
        report=json.loads((P/'preparation_report.json').read_text())
        for kind,(file,_) in FILES.items():
            self.assertEqual(hashlib.sha256((ROOT/'input'/file).read_bytes()).hexdigest(),report['sources'][kind]['sha256'])

if __name__=='__main__':unittest.main(verbosity=2)
