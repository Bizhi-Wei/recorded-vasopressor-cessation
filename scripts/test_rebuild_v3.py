import unittest
import numpy as np
import pandas as pd
from rebuild_v3_core import *


def synthetic(parts, hours=40):
    origin=pd.Timestamp('2150-01-01')
    records=[]
    for drug,start,end,rate in parts:
        records.append(dict(drug=drug,starttime=origin+pd.Timedelta(hours=start),endtime=origin+pd.Timedelta(hours=end),rate=rate,rateuom='units/hour' if drug==VA else 'mcg/kg/min',patientweight=80,statusdescription='FinishedRunning'))
    return build_timeline(pd.DataFrame(records),origin,origin+pd.Timedelta(hours=hours)),seconds(origin)


class CoreTests(unittest.TestCase):
    def test_datetime_resolution_and_stable(self):
        tl,o=synthetic([(NE,0,8,.05),(VA,0,8,2.4)])
        self.assertEqual(first_stable(tl,o),o+7200)
        self.assertEqual(at(tl,o+3600)[0],.05)
    def test_future_cannot_move_eligibility(self):
        x,o=synthetic([(NE,0,4,.05),(VA,0,8,2.4)])
        y,_=synthetic([(NE,0,4,.05),(VA,0,8,2.4),(NE,10,39,1.)])
        self.assertEqual(first_stable(x,o),first_stable(y,o))
    def test_surgery_not_future(self):
        tl,o=synthetic([(NE,0,30,.05),(VA,0,30,2.4)])
        self.assertGreaterEqual(first_stable(tl,o+24*3600),o+24*3600)
    def test_late_stable_state_scanned(self):
        tl,o=synthetic([(NE,0,9,.3),(NE,9,18,.05),(VA,0,18,2.4)])
        self.assertEqual(first_stable(tl,o),o+11*3600)
    def test_short_stop_and_restart(self):
        tl,o=synthetic([(NE,0,4,.05),(NE,4.1,20,.05),(VA,0,20,2.4)])
        action,drug=first_action(tl,o+7200,o+14*3600)
        self.assertEqual(drug,NE)
        self.assertEqual(action,o+4*3600)
        t,kind=rescue_time(tl,action,drug,o+20*3600)
        self.assertAlmostEqual(t,o+4.1*3600)
        self.assertEqual(kind,'restart')
    def test_no_post_observation_event(self):
        tl,o=synthetic([(NE,0,4,.05),(NE,6,20,.05),(VA,0,20,2.4)])
        self.assertTrue(np.isinf(rescue_time(tl,o+4*3600,NE,o+5*3600)[0]))
    def test_precise_epi_and_short_peak(self):
        tl,o=synthetic([(NE,0,4,.05),(VA,0,4,2.4),('epinephrine',2.01,2.02,.03)])
        mx,mean=summarize_ned(tl,o+7200,o+10800)
        self.assertAlmostEqual(mx,.18)
        self.assertAlmostEqual(mean,.1503)
    def test_deadline_and_strategy_labels(self):
        self.assertEqual(clone_censor(0,np.inf,'none',0),43200)
        self.assertTrue(np.isinf(clone_censor(0,43200,NE,0)))
        self.assertEqual(clone_censor(0,100,NE,1),100)
        self.assertTrue(np.isinf(clone_censor(0,100,VA,1)))
    def test_competing_risk(self):
        self.assertAlmostEqual(weighted_cif([1,2,3,4],[1,2,1,0],[1,1,1,1],24),.5)
    def test_unknown_is_not_zero(self):
        tl,o=synthetic([(NE,0,8,.05),(VA,0,8,2400)])
        self.assertIsNone(first_stable(tl,o))
        self.assertTrue(np.isnan(ned(tl,o+3600)))
    def test_concurrent_change_precedes_later_restart(self):
        tl,o=synthetic([(NE,0,4,.05),(NE,6,20,.05),(VA,0,4,2.4),(VA,4,20,4.)])
        self.assertEqual(rescue_time(tl,o+4*3600,NE,o+20*3600)[0],o+4*3600)
    def test_contiguous_dose_changes_not_withdrawal(self):
        tl,o=synthetic([(NE,0,4,.05),(NE,4,7,.04),(VA,0,10,2.4)])
        t,drug=first_action(tl,o+2*3600,o+9*3600)
        self.assertEqual(t,o+7*3600)
        self.assertEqual(drug,NE)
    def test_simultaneous_stop_does_not_prefer_ne(self):
        tl,o=synthetic([(NE,0,4,.05),(VA,0,4,2.4)])
        t,drug=first_action(tl,o+7200,o+10*3600)
        self.assertEqual(drug,'simultaneous')
        for arm in [0,1]:self.assertEqual(clone_censor(o+7200,t,drug,arm),t)
    def test_unknown_window_censors_at_its_start(self):
        tl,o=synthetic([(NE,0,10,.05),(VA,0,10,2.4),('epinephrine',4,5,99)])
        self.assertEqual(first_unknown(tl,o+7200,o+10*3600),o+4*3600)
    def test_no_restart_before_withdrawal(self):
        tl,o=synthetic([(NE,0,4,.05),(NE,5,20,.05),(VA,0,20,2.4)])
        t,kind=deterioration_time(tl,o+7200,o+4*3600,NE,o+10*3600)
        self.assertEqual(t,o+5*3600)
        self.assertEqual(kind,'post_action_restart')
    def test_prewithdrawal_deterioration_has_distinct_name(self):
        tl,o=synthetic([(NE,0,3,.05),(NE,3,4,.3),(VA,0,10,2.4)])
        t,kind=deterioration_time(tl,o+7200,o+4*3600,NE,o+10*3600)
        self.assertEqual(t,o+3*3600)
        self.assertEqual(kind,'ned_escalation')
    def test_paused_is_end_status_not_absent_delivery(self):
        o=pd.Timestamp('2150-01-01')
        rows=pd.DataFrame([dict(drug=NE,starttime=o,endtime=o+pd.Timedelta(hours=4),rate=.05,rateuom='mcg/kg/min',patientweight=80,statusdescription='Paused'),dict(drug=VA,starttime=o,endtime=o+pd.Timedelta(hours=8),rate=2.4,rateuom='units/hour',patientweight=80,statusdescription='FinishedRunning')])
        tl=build_timeline(rows,o,o+pd.Timedelta(hours=10))
        self.assertEqual(first_stable(tl,seconds(o)),seconds(o)+7200)
        self.assertEqual(first_action(tl,seconds(o)+7200,seconds(o)+9*3600)[0],seconds(o)+4*3600)


if __name__=='__main__':unittest.main()
