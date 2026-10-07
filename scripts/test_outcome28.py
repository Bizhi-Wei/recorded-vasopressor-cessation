import unittest
import pandas as pd
from outcome28_core import death_status, union_seconds

class OutcomeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.a=pd.Timestamp('2150-01-01 12:00'); self.z=self.a+pd.Timedelta(days=28)
        self.c=self.z+pd.Timedelta(days=365)
    def classify(self, exact=pd.NaT,dod=pd.NaT,coverage=None,conflict=False):
        return death_status(self.a,self.z,exact,dod,self.c if coverage is None else coverage,conflict)
    def test_exact_boundaries(self):
        self.assertEqual(self.classify(exact=self.a),'DEATH_AT_OR_BEFORE_ANCHOR')
        self.assertEqual(self.classify(exact=self.z),'DEATH_WITHIN28_EXACT')
        self.assertEqual(self.classify(exact=self.z+pd.Timedelta(seconds=1)),'NO_DEATH_WITHIN28_LATER_DEATH')
    def test_date_boundaries(self):
        self.assertEqual(self.classify(dod=self.a.normalize()),'DATE_AMBIGUOUS_AT_ANCHOR')
        self.assertEqual(self.classify(dod=self.z.normalize()),'DATE_AMBIGUOUS_AT_DAY28')
        self.assertEqual(self.classify(dod=self.z.normalize()-pd.Timedelta(days=1)),'DEATH_WITHIN28_DATE')
    def test_missing_and_conflicts(self):
        self.assertEqual(self.classify(),'NO_DEATH_RECORDED_WITHIN_LINKAGE_WINDOW')
        self.assertEqual(self.classify(coverage=self.z-pd.Timedelta(seconds=1)),'LINKAGE_WINDOW_INSUFFICIENT')
        self.assertEqual(self.classify(conflict=True),'RECORD_CONFLICT')
    def test_union_overlap_clip_gap(self):
        self.assertEqual(union_seconds([(-2,2),(1,4),(4,5),(7,20)],0,10),8)
        self.assertEqual(union_seconds([],0,10),0)
        self.assertEqual(union_seconds([(1,1),(4,2)],0,10),0)

if __name__=='__main__': unittest.main()
