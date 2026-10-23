import unittest
import numpy as np
import pandas as pd
from template_groups import build_template_groups
from train_category_pair import HERE

class TemplateTests(unittest.TestCase):
    def test_similarity_and_original_links_are_both_preserved(self):
        d=pd.DataFrame({'ticket_id':['a','b','c','d'],'model_text':['alpha','paraphrase','linked event','different'],'evaluation_group':['g1','g2','g2','g3']})
        X=np.array([[1.,0],[.999,.01],[0,1],[-1,0]])
        result=build_template_groups(d,X,.90)
        self.assertEqual(result.evaluation_group.nunique(),2)
        self.assertEqual(result.evaluation_group.iloc[0],result.evaluation_group.iloc[2])
        self.assertNotEqual(result.evaluation_group.iloc[0],result.evaluation_group.iloc[3])
    def test_no_label_dependency(self):
        d=pd.DataFrame({'ticket_id':['a','b'],'model_text':['x','y'],'evaluation_group':['g1','g2'],'pair_label':['A','B']})
        X=np.array([[1.,0],[1.,.01]])
        first=build_template_groups(d,X).evaluation_group
        d['pair_label']=['different','labels']
        self.assertEqual(first.tolist(),build_template_groups(d,X).evaluation_group.tolist())
    def test_template_oof_integrity(self):
        folder=HERE/'results_template_family'
        d=pd.read_csv(folder/'grouped_oof_predictions.csv')
        self.assertTrue(d.ticket_id.is_unique)
        self.assertEqual(d.groupby('evaluation_group').fold.nunique().max(),1)
        self.assertEqual(d.groupby('original_evaluation_group').fold.nunique().max(),1)
        a=pd.read_csv(folder/'outer_fold_assignments.csv')
        for _,part in a.groupby('fold'):
            self.assertTrue(set(part[part.role=='train'].evaluation_group).isdisjoint(part[part.role=='test'].evaluation_group))
        links=pd.read_csv(folder/'similarity_links.csv')
        lookup=d.set_index('ticket_id').fold
        self.assertTrue((links.ticket_a.map(lookup)==links.ticket_b.map(lookup)).all())

if __name__=='__main__':unittest.main()
