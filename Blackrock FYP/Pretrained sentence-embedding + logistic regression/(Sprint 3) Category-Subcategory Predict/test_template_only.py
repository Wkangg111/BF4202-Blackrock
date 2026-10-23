import unittest
import numpy as np
import pandas as pd
from template_groups import build_template_groups
from train_category_pair import HERE

class TemplateOnlyTests(unittest.TestCase):
    def test_event_links_ignored_but_duplicates_preserved(self):
        d=pd.DataFrame({'ticket_id':['a','b','c'],'model_text':['first','unrelated','first'],'evaluation_group':['same','same','same']})
        X=np.array([[1.,0.],[0.,1.],[1.,0.]])
        r=build_template_groups(d,X,preserve_event_links=False)
        self.assertEqual(r.evaluation_group.iloc[0],r.evaluation_group.iloc[2])
        self.assertNotEqual(r.evaluation_group.iloc[0],r.evaluation_group.iloc[1])
    def test_held_out_groups_and_similarity_links(self):
        folder=HERE/'results_template_only'
        d=pd.read_csv(folder/'grouped_oof_predictions.csv')
        original=pd.read_csv(HERE/'results/grouped_oof_predictions.csv')
        self.assertEqual(set(d.ticket_id),set(original.ticket_id))
        self.assertTrue(d.ticket_id.is_unique)
        self.assertEqual(d.groupby('evaluation_group').fold.nunique().max(),1)
        self.assertEqual(d.groupby('model_text').fold.nunique().max(),1)
        lookup=d.set_index('ticket_id').fold
        edges=pd.read_csv(folder/'similarity_links.csv')
        self.assertTrue((edges.ticket_a.map(lookup)==edges.ticket_b.map(lookup)).all())
        a=pd.read_csv(folder/'outer_fold_assignments.csv')
        for _,part in a.groupby('fold'):
            tr=part[part.role=='train'];te=part[part.role=='test']
            self.assertTrue(set(tr.evaluation_group).isdisjoint(te.evaluation_group))

if __name__=='__main__':unittest.main()
