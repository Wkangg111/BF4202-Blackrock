import unittest
import pandas as pd
from prepare_combined_risk import combined_labels
class CombinedTests(unittest.TestCase):
 def test_labels_and_unknown_timing(self):
  t=pd.DataFrame({'ticket_id':['A','B','C','D','E','F'],'created_at':['2024-01-01']*6,
   'closed_at':['2024-01-03']*6,'label_known_at':['2024-01-03']*6,
   'escalation_label':[1,0,0,None,None,0]})
  t.loc[[3,4],'closed_at']=None;t.loc[[3,4],'label_known_at']=None
  links=pd.DataFrame({'ticket_id':['A','B','D','F'],'start_at':['2024-01-01']*3+['2025-01-01']})
  r=combined_labels(t,links,pd.Timestamp('2024-02-01')).set_index('ticket_id')
  self.assertEqual(r.loc['A','risk_label'],1)
  self.assertEqual(r.loc['B','risk_label'],1)
  self.assertEqual(r.loc['C','risk_label'],0)
  self.assertEqual(r.loc['D','risk_label'],1)
  self.assertTrue(pd.isna(r.loc['E','risk_label']))
  self.assertTrue(pd.isna(r.loc['F','risk_label']))
  self.assertTrue(pd.isna(r.loc['B','risk_label_known_at']))
  self.assertFalse(r.loc['E','label_timing_verified'])
 def test_duplicate_links_do_not_multiply_tickets(self):
  t=pd.DataFrame({'ticket_id':['A'],'created_at':['2024-01-01'],'closed_at':[None],'label_known_at':[None],'escalation_label':[None]})
  l=pd.DataFrame({'ticket_id':['A','A'],'start_at':['2024-01-01']*2})
  self.assertEqual(len(combined_labels(t,l,pd.Timestamp('2024-02-01'))),1)
if __name__=='__main__':unittest.main()
