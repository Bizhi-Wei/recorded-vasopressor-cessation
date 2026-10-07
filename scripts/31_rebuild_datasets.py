"""Create source-timed covariates, exact endpoints and CCW risk-set data."""
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from rebuild_v3_core import *

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'rebuild_v3'
LAB={50813:'lactate',50912:'creatinine',51265:'platelets',51300:'wbc',51301:'wbc',51222:'hemoglobin',51006:'bun'}
MAP=[220052,220181,225312]
PHYS=['map_mean','map_min','hr_mean','lactate','creatinine','platelets','wbc','hemoglobin','bun','urine_6h']
BASE_COV=['age','male','weight','cabg','valve','cpb','hours_icu','hours_surgery','overlap_before_h','ne','va','ned','rrt_before','mcs_before']+PHYS


class FeatureReader:
    def __init__(self):
        self.chart={k:v.sort_values('charttime') for k,v in pd.read_pickle(OUT/'raw_chart.pkl').groupby('stay_id')}
        self.lab={k:v.sort_values(['charttime','storetime']) for k,v in pd.read_pickle(OUT/'raw_lab.pkl').groupby('hadm_id')}
        self.urine={k:v for k,v in pd.read_pickle(OUT/'raw_urine.pkl').groupby('stay_id')}
        pe=pd.read_pickle(OUT/'raw_procedure.pkl')
        di=pd.read_csv(BASE.parent/'icu/d_items.csv',usecols=['itemid','label'])
        pe=pe.merge(di,on='itemid',how='left')
        pe['rrt']=pe.label.str.contains('Dialysis|CRRT|CVVH',case=False,na=False)
        pe['mcs']=pe.label.str.contains('IABP|Impella|ECMO|Balloon Pump|VAD',case=False,na=False)
        self.pe={k:v for k,v in pe.groupby('stay_id')}
        self.pr={k:v.sort_values('starttime') for k,v in pd.read_pickle(OUT/'source_pressors.pkl').groupby('stay_id')}
        self.source_assertions=0

    def features(self,r,tl,t):
        dt=timestamp(t)
        before=at(tl,t,True)
        lo,hi,v=window(tl,tl['bounds'][0],t)
        both=np.isfinite(v[:,:2]).all(axis=1)&(v[:,:2]>0).all(axis=1)
        rec={k:getattr(r,k) for k in ['age','male','cabg','valve','cpb']}
        rec.update(hours_icu=(t-seconds(r.intime))/3600,hours_surgery=(t-seconds(r.surgery_date))/3600,
            overlap_before_h=float(((hi-lo)[both]).sum()/3600),ne=before[0],va=before[1],ned=float(before@COEF))
        pr=self.pr.get(r.stay_id)
        ww=pr[(pr.starttime<dt)&pr.patientweight.between(20,300)] if pr is not None else pd.DataFrame()
        rec['weight']=float(ww.patientweight.iloc[-1]) if len(ww) else np.nan
        rec.update({k:np.nan for k in PHYS})
        chart=self.chart.get(r.stay_id)
        if chart is not None:
            s=chart[(chart.charttime<dt)&(chart.charttime>=dt-pd.Timedelta(hours=6))]
            mv=s[s.itemid.isin(MAP)&s.valuenum.between(20,200)].valuenum
            hv=s[(s.itemid==220045)&s.valuenum.between(20,250)].valuenum
            rec.update(map_mean=mv.mean(),map_min=mv.min(),hr_mean=hv.mean())
            assert not len(s) or s.charttime.max()<dt
        lab=self.lab.get(r.hadm_id)
        if lab is not None:
            s=lab[(lab.charttime<dt)&(lab.storetime<dt)&(lab.charttime>=dt-pd.Timedelta(hours=24))&lab.valuenum.notna()]
            for item,name in LAB.items():
                # Joint latest value for both WBC item identifiers.
                ids=[i for i,n in LAB.items() if n==name]
                vals=s[s.itemid.isin(ids)&(s.valuenum>0)]
                if name=='lactate':vals=vals[(vals.valuenum<40)&(vals.charttime>=dt-pd.Timedelta(hours=6))]
                if len(vals):rec[name]=float(vals.valuenum.iloc[-1])
            assert not len(s) or max(s.charttime.max(),s.storetime.max())<dt
        urine=self.urine.get(r.stay_id)
        if urine is not None:
            s=urine[(urine.charttime<dt)&(urine.storetime<dt)&(urine.charttime>=dt-pd.Timedelta(hours=6))&urine.value.between(0,5000)]
            rec['urine_6h']=float(s.value.sum()) if len(s) else np.nan
            assert not len(s) or max(s.charttime.max(),s.storetime.max())<dt
        pe=self.pe.get(r.stay_id)
        for name in ['rrt','mcs']:
            rec[name+'_before']=int(((pe.starttime<dt)&pe[name]).any()) if pe is not None else 0
        self.source_assertions+=1
        return rec

    def outcomes_phys(self,r,a,end):
        rec={}
        chart=self.chart.get(r.stay_id)
        lab=self.lab.get(r.hadm_id)
        for low,high,tag in [(0,6,'0_6'),(6,12,'6_12'),(12,24,'12_24')]:
            z=min(a+high*3600,end)
            t=a+low*3600
            rec[f'observed_h_{tag}']=max(0,z-t)/3600
            rec[f'map_{tag}']=rec[f'hypo65_{tag}']=rec[f'lactate_{tag}']=np.nan
            rec[f'n_map_{tag}']=rec[f'n_lactate_{tag}']=0
            if z<=t:continue
            if chart is not None:
                s=chart[(chart.charttime>=timestamp(t))&(chart.charttime<timestamp(z))&chart.itemid.isin(MAP)&chart.valuenum.between(20,200)]
                rec[f'n_map_{tag}']=len(s)
                if len(s):
                    rec[f'map_{tag}']=s.valuenum.mean()
                    rec[f'hypo65_{tag}']=(s.valuenum<65).mean()
            if lab is not None:
                s=lab[(lab.charttime>=timestamp(t))&(lab.charttime<timestamp(z))&(lab.itemid==50813)&lab.valuenum.between(.01,39.99)]
                rec[f'n_lactate_{tag}']=len(s)
                if len(s):rec[f'lactate_{tag}']=float(s.valuenum.iloc[-1])
        return rec


def terminal(t0,event,death,icuout,unknown,mode):
    """Clinical event wins ties with competing death/discharge; unknown is censoring."""
    target=min(event,death) if mode=='ccw' else event
    competing=icuout if mode=='ccw' else min(death,icuout)
    stop=min(target,competing,unknown,t0+86400)
    if target<=stop:eventcode=1
    elif competing<=stop:eventcode=2
    else:eventcode=0
    return stop,eventcode


def main():
    pool=pd.read_csv(OUT/'stable_pool.csv')
    for c in ['stable_t0','action_time','intime','outtime','deathtime','surgery_date','surgery_available']:
        pool[c]=pd.to_datetime(pool[c],errors='coerce')
    tlmap=pickle.load((OUT/'timelines.pkl').open('rb'))
    reader=FeatureReader()
    decisions=[];clones=[];blocks=[];excluded=[]
    for number,r in enumerate(pool.itertuples(),1):
        tl=tlmap[r.stay_id];t0=seconds(r.stable_t0)
        death=seconds(r.deathtime) if pd.notna(r.deathtime) else np.inf
        icuout=seconds(r.outtime)
        action=seconds(r.action_time) if pd.notna(r.action_time) else np.inf
        unknown=first_unknown(tl,t0,min(t0+86400,death,icuout))
        if not np.isfinite(ned(tl,t0,True)) or unknown<=t0:
            excluded.append(dict(stay_id=r.stay_id,design='ccw',reason='unknown_rate_at_time_zero'))
            continue
        base=reader.features(r,tl,t0)
        event,reason=deterioration_time(tl,t0,action,r.first_drug,min(t0+86400,death,icuout,unknown))
        end,code=terminal(t0,event,death,icuout,unknown,'ccw')
        for arm in [0,1]:
            censor=clone_censor(t0,action,r.first_drug,arm)
            stop=min(end,censor)
            term=code if end<=censor else 3 # target / competitor before-or-at censor
            clones.append(dict(stay_id=r.stay_id,subject_id=r.subject_id,arm=arm,t0=t0,stop_h=(stop-t0)/3600,
                terminal=term,censor_h=(censor-t0)/3600,event_reason=reason,action_h=(action-t0)/3600,**base))
            for k in range(24):
                a=t0+3600*k;z=a+3600
                if stop<a or (stop==a and k>0):break
                done=(r.first_drug==(VA if arm else NE)) and (action<=a if k>0 else action<a)
                censor_now=int(term==3 and stop<=z)
                target_now=int(term==1 and stop<=z)
                compete_now=int(term==2 and stop<=z)
                feat=reader.features(r,tl,a) if k else base
                blocks.append(dict(stay_id=r.stay_id,subject_id=r.subject_id,arm=arm,block=k,
                    pending=int(not done and k<12),censor=censor_now,target=target_now,compete=compete_now,
                    obs_censor=int(term==0 and stop<z),stop_h=(stop-t0)/3600,**feat))
                if stop<=z:break
        if r.first_drug not in [NE,VA] or not np.isfinite(action):continue
        # A decision after unknown earlier observation is not a reliable first action.
        if action>=min(death,icuout,unknown):
            excluded.append(dict(stay_id=r.stay_id,design='decision',reason='no_reliable_action_before_observation_end'))
            continue
        ua=first_unknown(tl,action,min(action+86400,death,icuout))
        if not np.isfinite(ned(tl,action,True)) or ua<=action:continue
        event,reason=rescue_time(tl,action,r.first_drug,min(action+86400,death,icuout,ua))
        end,code=terminal(action,event,death,icuout,ua,'decision')
        obs=min(death,icuout,ua,action+86400)
        rec=dict(stay_id=r.stay_id,subject_id=r.subject_id,hadm_id=r.hadm_id,A=int(r.first_drug==VA),
            stable_t0=r.stable_t0,t0=timestamp(action),time_h=(end-action)/3600,event=code,event_reason=reason,
            observed_h=(obs-action)/3600,unknown_censor=int(ua<=min(death,icuout,action+86400)),
            **reader.features(r,tl,action))
        for W in [6,12,24]:
            mx,mn=summarize_ned(tl,action,min(action+W*3600,obs))
            rec[f'dned_peak_{W}h']=mx-rec['ned'];rec[f'dned_mean_{W}h']=mn-rec['ned']
            rec[f'full_{W}h']=int(obs>=action+W*3600)
        for low,high in [(0,6),(6,12),(12,24)]:
            mx,mn=summarize_ned(tl,action+low*3600,min(action+high*3600,obs))
            rec[f'ned_mean_{low}_{high}']=mn
        rec.update(reader.outcomes_phys(r,action,obs))
        decisions.append(rec)
        if number%50==0:print('datasets processed',number,'of',len(pool),flush=True)
    d=pd.DataFrame(decisions);c=pd.DataFrame(clones);long=pd.DataFrame(blocks)
    d.to_csv(OUT/'decision_analysis.csv',index=False)
    c.to_csv(OUT/'ccw_clones.csv',index=False)
    long.to_csv(OUT/'ccw_long.csv',index=False)
    pd.DataFrame(excluded).to_csv(OUT/'observation_exclusions.csv',index=False)
    checks=dict(decision_subset=bool(set(d.stay_id).issubset(pool.stay_id)),
        decision_after_stable=bool((pd.to_datetime(d.t0)>=pd.to_datetime(d.stable_t0)).all()),
        decision_no_negative_time=bool((d.time_h>=0).all()),ccw_no_duplicate_blocks=not long.duplicated(['stay_id','arm','block']).any(),
        ccw_two_clones_each=bool(c.groupby('stay_id').size().eq(2).all()),
        exact_deadline_censor_accounted=bool(c[(c.terminal==3)&c.stop_h.eq(12)].shape[0]==long[(long.censor==1)&long.block.eq(11)&long.stop_h.eq(12)].shape[0]),
        no_future_features=reader.source_assertions>0)
    assert all(checks.values()),checks
    report=dict(decision_n=len(d),decision_groups=d.A.value_counts().to_dict(),decision_events=d.event.value_counts().to_dict(),
        unknown_censor_n=int(d.unknown_censor.sum()),decision_full_windows={str(W):int(d[f'full_{W}h'].sum()) for W in [6,12,24]},
        ccw_pool_n=c.stay_id.nunique(),ccw_clones=len(c),ccw_rows=len(long),
        ccw_terminals=c.groupby(['arm','terminal']).size().to_dict().__str__(),
        feature_time_checks=reader.source_assertions,checks=checks,covariates=BASE_COV,status='DATASETS_BUILT_NOT_YET_INFERRED')
    (OUT/'dataset_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
