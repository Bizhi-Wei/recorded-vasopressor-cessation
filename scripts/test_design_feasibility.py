import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('design37',Path(__file__).with_name('37_design_feasibility.py'))
design=importlib.util.module_from_spec(spec);spec.loader.exec_module(design)

def timeline(bounds,ne):
    rates=np.zeros((len(ne),6));rates[:,0]=ne;rates[:,1]=2.4
    return {'bounds':np.asarray(bounds,float),'rates':rates}

class DesignTests(unittest.TestCase):
    def test_incomplete_observation_not_no_restart(self):
        tl=timeline([0,600],[0])
        self.assertEqual(design.restart_observation(tl,0,0,120,15),(None,None))

    def test_restart_at_horizon(self):
        tl=timeline([0,900,1200],[0,.05])
        self.assertEqual(design.restart_observation(tl,0,0,1200,15),(1,15.))

    def test_no_restart_after_observation_end(self):
        tl=timeline([0,300,1200],[0,.05])
        self.assertEqual(design.restart_observation(tl,0,0,300,15),(None,None))

    def test_stability_requires_full_past_two_hours(self):
        self.assertFalse(design.past_stable(timeline([0,3600],[.05]),3600))
        self.assertTrue(design.past_stable(timeline([0,7200],[.05]),7200))
        self.assertFalse(design.past_stable(timeline([0,3600,7200],[.2,.05]),7200))

if __name__=='__main__':unittest.main()
