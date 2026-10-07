from pathlib import Path
import importlib.util
import unittest
import numpy as np
from rebuild_v3_core import weighted_cif

spec=importlib.util.spec_from_file_location('overlap38',Path(__file__).with_name('38_overlap_analysis.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class OverlapTests(unittest.TestCase):
    def test_strategy_weight_direction(self):
        np.testing.assert_allclose(m.overlap_weights([0,1],[.2,.8]),[.2,.2])
    def test_other_first_components_are_competing(self):
        code=m.component_code([1,1,2,0],['restart','escalation','none','none'],'restart')
        np.testing.assert_array_equal(code,[1,2,2,0])
    def test_component_cif_sum_with_censor_and_competitor(self):
        time=np.array([1,2,3,4,5]);event=np.array([0,1,2,1,0]);w=np.array([2,1,3,2,1])
        reasons=np.array(['none','restart','none','escalation','none'])
        parts=[weighted_cif(time,m.component_code(event,reasons,p),w,24) for p in m.PARTS]
        self.assertAlmostEqual(sum(parts),weighted_cif(time,event,w,24))
    def test_weighted_cif_arm_scale_invariance(self):
        self.assertAlmostEqual(weighted_cif([1,2,3],[1,2,0],[.1,.2,.4],24),weighted_cif([1,2,3],[1,2,0],[1,2,4],24))

if __name__=='__main__':unittest.main()
