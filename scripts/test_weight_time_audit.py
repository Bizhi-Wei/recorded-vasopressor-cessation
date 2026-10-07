"""Hand-computable competing-risk and time-grid regression cases."""
from pathlib import Path
import importlib.util
import unittest
import numpy as np
import pandas as pd

def load(name,filename):
    path=Path(__file__).parent/filename
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

build=load('build34','34_weight_time_build.py')
infer=load('infer35','35_weight_time_inference.py')

def example(minutes,cases):
    rows=[]
    for arm in [0,1]:
        for i,(stop,terminal) in enumerate(cases):
            for row in build.grid_rows(stop,terminal,np.inf,False,minutes):
                rows.append(dict(arm=arm,subject_id=i,stay_id=i,clone_id=f'{i}_{arm}',**row))
    return pd.DataFrame(rows)

class AuditTests(unittest.TestCase):
    def test_deadline_retained_at_both_resolutions(self):
        for m,k in [(60,11),(15,47)]:
            rows=list(build.grid_rows(12,3,np.inf,False,m))
            self.assertEqual(rows[-1]['block'],k)
            self.assertEqual(rows[-1]['censor'],1)
            self.assertEqual(sum(x['censor'] for x in rows),1)

    def test_boundary_adherence_stops_probability_updates(self):
        rows=list(build.grid_rows(24,0,1,True,15))
        self.assertEqual([x['pending'] for x in rows[:5]],[1,1,1,1,0])
        self.assertTrue(all(x['pending']==0 for x in rows[4:]))

    def test_time_zero_event_is_kept(self):
        rows=list(build.grid_rows(0,1,0,True,15))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['target'],1)

    def test_competing_cif_hand_calculation(self):
        for m in [60,15]:
            d=example(m,[(2,1),(3,2),(24,0)])
            risk,_=infer.weighted_hazards(d,np.ones(len(d)),m)
            np.testing.assert_allclose(risk,[1/3,1/3])

    def test_weights_scale_invariance(self):
        d=example(15,[(2,1),(3,2),(24,0)])
        w=np.where(d.subject_id.eq(0),2.,1.)
        r,_=infer.weighted_hazards(d,w,15)
        r2,_=infer.weighted_hazards(d,w*100,15)
        np.testing.assert_allclose(r,[.5,.5]);np.testing.assert_allclose(r,r2)

    def test_same_bin_censoring_is_explicit_approximation(self):
        # A person censored at 1.75h is still at risk at a 1.25h event.
        # Coarse complete-bin analysis drops that person early; fine grid does not.
        risks=[]
        for m in [60,15]:
            d=example(m,[(1.25,1),(1.75,3),(24,0)])
            risks.append(infer.weighted_hazards(d,np.ones(len(d)),m)[0][0])
        np.testing.assert_allclose(risks,[.5,1/3])

    def test_cluster_resampling_keeps_independent_copies(self):
        d=example(60,[(2,1),(3,2),(24,0)])
        q=infer.resample(d,infer.sampler(d),[0,0,2])
        self.assertEqual(q.clone_id.nunique(),6)
        self.assertEqual(q[q.subject_id==0].clone_id.nunique(),4)
        self.assertFalse(q.duplicated(['clone_id','block']).any())

if __name__=='__main__':unittest.main()
