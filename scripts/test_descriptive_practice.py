from pathlib import Path
import importlib.util
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('practice43',Path(__file__).with_name('43_descriptive_practice.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class PracticeTests(unittest.TestCase):
    def test_time_weighted_not_row_average(self):
        tl={'bounds':np.array([0.,60.,1200.]),'rates':np.array([[2.,1.],[0.,1.]])}
        r=m.summarize_window(tl,0,1200,0)
        self.assertAlmostEqual(r['mean'],.1);self.assertEqual(r['full_window'],1)
    def test_partial_window_not_filled_with_zero(self):
        tl={'bounds':np.array([600.,1200.]),'rates':np.array([[2.,1.]])}
        r=m.summarize_window(tl,0,1200,0)
        self.assertEqual(r['full_window'],0);self.assertEqual(r['mean'],2.)
    def test_unknown_excluded_from_known_integral(self):
        tl={'bounds':np.array([0.,600.,1200.]),'rates':np.array([[np.nan,1.],[2.,1.]])}
        r=m.summarize_window(tl,0,1200,0)
        self.assertEqual(r['full_window'],0);self.assertEqual(r['mean'],2.)
        self.assertAlmostEqual(r['unknown_h'],1/6)
    def test_net_change_boundary_and_incomplete(self):
        self.assertEqual(m.net_category(1,.9,True),'approximately_flat')
        self.assertEqual(m.net_category(1,.89,True),'net_down')
        self.assertEqual(m.net_category(0,1,True),'zero_at_window_start')
        self.assertEqual(m.net_category(1,.5,False),'incomplete')

if __name__=='__main__':unittest.main()
