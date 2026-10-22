import unittest,json
from pathlib import Path
import pandas as pd
import joblib
from train_category_pair import prepare_data,pair_label,HERE

class PairTests(unittest.TestCase):
    def test_pair_roundtrip(self):
        self.assertEqual(json.loads(pair_label('A | B','Quotes " and comma,')), ['A | B','Quotes " and comma,'])
    def test_group_integrity(self):
        d,_=prepare_data()
        self.assertTrue(d.ticket_id.is_unique)
        for key in ['model_text','event_group_id']:
            self.assertLessEqual(d.groupby(key).evaluation_group.nunique().max(),1)
    def test_held_out_predictions_and_saved_model(self):
        out=HERE/'results';d,_=prepare_data()
        pred=pd.read_csv(out/'grouped_oof_predictions.csv')
        self.assertEqual(set(d.ticket_id),set(pred.ticket_id));self.assertTrue(pred.ticket_id.is_unique)
        self.assertEqual(set(pred.fold),{1,2,3,4,5})
        self.assertEqual(pred.groupby('evaluation_group').fold.nunique().max(),1)
        a=pd.read_csv(out/'outer_fold_assignments.csv')
        for fold,rows in a.groupby('fold'):
            tr=rows[rows.role=='train'];te=rows[rows.role=='test']
            self.assertTrue(set(tr.evaluation_group).isdisjoint(te.evaluation_group))
            pairs=set(d[d.ticket_id.isin(tr.ticket_id)].pair_label)
            self.assertTrue(set(pred[pred.fold==fold].predicted_pair)<=pairs)
        artifact=joblib.load(out/'category_pair_model.joblib')
        self.assertEqual(set(artifact['classifier'].classes_),set(d.pair_label))

if __name__=='__main__':unittest.main()
