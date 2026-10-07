from pathlib import Path
import importlib.util
import unittest
import numpy as np
import pandas as pd
spec=importlib.util.spec_from_file_location('nonlinear41',Path(__file__).with_name('41_nonlinear_dose_diagnostics.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class NonlinearTests(unittest.TestCase):
    def test_fixed_knots_not_changed_by_other_samples(self):
        d=pd.DataFrame({'ne':[.02,.05],'va':[1.2,2.4]})
        a=m.spline_columns(d)
        extended=pd.concat([d,pd.DataFrame({'ne':[1.4],'va':[10.]})],ignore_index=True)
        np.testing.assert_allclose(a,m.spline_columns(extended).iloc[:2],atol=1e-12)
    def test_basis_handles_missing_and_extrapolation(self):
        b=m.spline_columns(pd.DataFrame({'ne':[np.nan,.05,2.],'va':[np.nan,2.4,15.]}))
        self.assertTrue(np.isfinite(b.to_numpy()).all())
        self.assertEqual(b.shape,(3,14))
    def test_negative_dose_rejected(self):
        with self.assertRaises(ValueError):m.spline_columns(pd.DataFrame({'ne':[-.1,.1],'va':[1.,2.]}))
    def test_weighted_quantile_hand_example(self):
        self.assertEqual(m.weighted_quantile([3,1,2],[1,1,8],.5),2)
    def test_equal_mean_not_equal_distribution(self):
        x=pd.DataFrame({'dose':[0.,2.,1.,1.]});a=np.array([0,0,1,1])
        b=m.ow.distribution_balance(x,a,np.ones(4),True).iloc[0]
        self.assertAlmostEqual(b.smd,0.)
        self.assertAlmostEqual(b.ks,.5)

if __name__=='__main__':unittest.main()
