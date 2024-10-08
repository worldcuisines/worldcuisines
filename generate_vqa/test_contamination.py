import pandas as pd

from utils import *

import unittest
import numpy as np

class TestContamination(unittest.TestCase):
    def test_eval_is_subset(self):
        s_t1 = pd.read_csv(os.path.join(RESOURCE_DIR, "small_eval_task1.csv"))
        s_t2 = pd.read_csv(os.path.join(RESOURCE_DIR, "small_eval_task2.csv"))
        l_t1 = pd.read_csv(os.path.join(RESOURCE_DIR, "large_eval_task1.csv"))
        l_t2 = pd.read_csv(os.path.join(RESOURCE_DIR, "large_eval_task2.csv"))
        
        assert(set(s_t1['food_id'].unique()).issubset(set(l_t1['food_id'].unique())))
        assert(set(s_t2['food_id'].unique()).issubset(set(l_t2['food_id'].unique())))
        assert(set(s_t1['prompt_id'].unique()).issubset(set(l_t1['prompt_id'].unique())))
        assert(set(s_t2['prompt_id'].unique()).issubset(set(l_t2['prompt_id'].unique())))
        assert(set(s_t1['food_id'].unique()) == (set(s_t2['food_id'].unique())))
        assert(set(l_t1['food_id'].unique()) == (set(l_t2['food_id'].unique())))
        assert(set(s_t1['prompt_id'].unique()) == (set(s_t2['prompt_id'].unique())))
        assert(set(l_t1['prompt_id'].unique()) == (set(l_t2['prompt_id'].unique())))
        
    def test_eval_disjoint_train(self):
        s_t1 = pd.read_csv(os.path.join(RESOURCE_DIR, "small_eval_task1.csv"))
        s_t2 = pd.read_csv(os.path.join(RESOURCE_DIR, "small_eval_task2.csv"))
        l_t1 = pd.read_csv(os.path.join(RESOURCE_DIR, "large_eval_task1.csv"))
        l_t2 = pd.read_csv(os.path.join(RESOURCE_DIR, "large_eval_task2.csv"))
        tr_t1 = pd.read_csv(os.path.join(RESOURCE_DIR, "train_task1.csv"))
        tr_t2 = pd.read_csv(os.path.join(RESOURCE_DIR, "train_task2.csv"))
        
        for ev in [s_t1, s_t2, l_t1, l_t2]:
            for tr in [tr_t1, tr_t2]:
                for col in ['food_id', 'prompt_id']:
                    unique_ev = set(ev[col].unique())
                    unique_tr = set(tr[col].unique())
                    assert(unique_ev.isdisjoint(unique_tr))
    
if __name__ == '__main__':
    unittest.main()

    