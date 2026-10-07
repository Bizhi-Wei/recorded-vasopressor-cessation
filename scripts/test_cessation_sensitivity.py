import unittest
import numpy as np
import pandas as pd
from cessation_sensitivity_core import bridge_gaps, candidates, prepare_rows, ending_source
from rebuild_v3_core import NE, VA

def timeline(bounds, ne, va):
    v = np.zeros((len(ne), 6)); v[:, 0] = ne; v[:, 1] = va
    return dict(bounds=np.array(bounds, dtype=float), rates=v)

class MeasurementTests(unittest.TestCase):
    def test_exact_threshold_and_no_mutation(self):
        t = timeline([0, 100, 400, 500], [1, 0, 1], [1, 1, 1]); old=t['rates'].copy()
        v,g,_=bridge_gaps(t,5,0,450,500)
        self.assertEqual(len(g),1); self.assertEqual(list(candidates(v,0,450)),[])
        np.testing.assert_array_equal(t['rates'],old)
        self.assertEqual(len(list(candidates(bridge_gaps(t,4.99,0,450,500)[0],0,450))),1)

    def test_unknown_not_bridged(self):
        t=timeline([0,100,200,300,500],[1,0,np.nan,1],[1,1,1,1])
        v,g,_=bridge_gaps(t,5,0,450,500)
        self.assertEqual(len(g),0); self.assertTrue(np.isnan(v['rates'][2,0]))

    def test_terminal_and_observation_boundary(self):
        t=timeline([0,100,200,500],[1,0,1],[1,1,1])
        v,g,u=bridge_gaps(t,5,0,450,200)
        self.assertEqual(len(g),0); self.assertEqual(len(u),1)
        self.assertEqual(next(candidates(v,0,200))[0],100)

    def test_leading_gap_not_bridged(self):
        t=timeline([0,100,300],[0,1],[1,1])
        v,g,_=bridge_gaps(t,5,0,250,300)
        self.assertEqual(v['rates'][0,0],0); self.assertEqual(len(g),0)

    def test_order_switch_and_future_lookahead(self):
        t=timeline([0,100,200,400,800],[1,0,0,1],[1,1,0,0])
        self.assertEqual(next(candidates(t,0,300))[1],NE)
        v,g,_=bridge_gaps(t,5,0,300,800)
        self.assertEqual(next(candidates(v,0,300))[1],VA)
        self.assertTrue(g[0]['restart_after_deadline'])

    def test_inclusive_deadline_exclusive_icu_end(self):
        t=timeline([0,100,200],[1,0],[1,1])
        self.assertEqual(next(candidates(t,0,100))[0],100)
        self.assertEqual(list(candidates(t,101,200)),[])

    def test_shifted_epoch_deadline_matches_reference(self):
        from rebuild_v3_core import first_action
        t=timeline([0,100,200],[1,0],[1,1]);t['bounds']+=6e9
        self.assertEqual(list(candidates(t,6e9,6e9+100)),[])
        self.assertEqual(first_action(t,6e9,6e9+100)[1],'none')

    def test_paused_source_and_tie(self):
        rows=[]
        for status in ['Paused','Stopped']:
            rows.append(dict(drug=NE,starttime=pd.Timestamp(0),endtime=pd.Timestamp(100,unit='s'),
                rate=.05,rateuom='mcg/kg/min',patientweight=80,statusdescription=status,orderid=1,linkorderid=1))
        r=ending_source(prepare_rows(pd.DataFrame(rows)),NE,100,.05)
        self.assertTrue(r['paused']); self.assertTrue(r['status_tie_disagreement'])
        self.assertEqual(r['status'],'Stopped')

    def test_overlap_latest_start_wins(self):
        rows=[]
        for start,rate,status in [(0,.05,'Paused'),(50,.02,'Stopped')]:
            rows.append(dict(drug=NE,starttime=pd.Timestamp(start,unit='s'),endtime=pd.Timestamp(100,unit='s'),
                rate=rate,rateuom='mcg/kg/min',patientweight=80,statusdescription=status,orderid=1,linkorderid=1))
        r=ending_source(prepare_rows(pd.DataFrame(rows)),NE,100,.02)
        self.assertFalse(r['paused']); self.assertEqual(r['normalized_rate'],.02)

    def test_paused_skip_requires_actual_dual_delivery(self):
        t=timeline([0,100,200,300,400],[1,0,1,1],[1,1,1,0])
        events=list(candidates(t,0,350))
        self.assertEqual([(e[0],e[1]) for e in events],[(100,NE),(300,VA)])
        accepted=[e for e in events if e[0]!=100]
        self.assertEqual(accepted[0][1],VA)
        t['rates'][2,0]=0
        self.assertEqual(len(list(candidates(t,0,350))),1)

if __name__=='__main__': unittest.main()
