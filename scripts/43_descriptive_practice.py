"""Unweighted practice descriptions with exact dose time integration and coverage."""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
from rebuild_v3_core import seconds,window,at,first_unknown,NE,VA

BASE=Path(__file__).resolve().parents[1];SRC=BASE/'rebuild_v3';OUT=SRC/'descriptive_practice_v1'
WINDOWS=[(-6,-4),(-4,-2),(-2,0)]

def summarize_window(tl,a,z,j):
    lo,hi,v=window(tl,a,z);dur=hi-lo
    known=np.isfinite(v[:,j]);covered=float(dur.sum());known_s=float(dur[known].sum())
    complete=abs(covered-(z-a))<1e-6 and known.all()
    return dict(mean=float(np.average(v[known,j],weights=dur[known])) if known_s>0 else np.nan,
        observed_h=covered/3600,known_h=known_s/3600,unknown_h=(covered-known_s)/3600,
        full_window=int(complete),start_rate=float(v[0,j]) if len(v) else np.nan,
        end_rate=float(v[-1,j]) if len(v) else np.nan)

def net_category(start,end,complete):
    if not complete or not np.isfinite(start+end):return 'incomplete'
    if start<=0:return 'zero_at_window_start'
    ratio=end/start
    return 'net_up' if ratio>1.1+1e-12 else ('net_down' if ratio<.9-1e-12 else 'approximately_flat')

def main():
    OUT.mkdir(exist_ok=True)
    inputs=[SRC/'stable_pool.csv',SRC/'timelines.pkl',SRC/'design_feasibility/decision_measurement_diagnostics.csv',
        BASE/'docs/descriptive_practice_plan_v1.md',BASE/'scripts/rebuild_v3_core.py',Path(__file__)]
    manifest=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs]
    (OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    pool=pd.read_csv(SRC/'stable_pool.csv')
    for col in ['stable_t0','action_time','intime','outtime','deathtime']:pool[col]=pd.to_datetime(pool[col])
    timelines=pickle.load((SRC/'timelines.pkl').open('rb'))
    states=[];person=[];segments=[];trajectory=[]
    for r in pool.itertuples():
        tl=timelines[r.stay_id];stable=seconds(r.stable_t0);deadline=stable+43200
        death=seconds(r.deathtime) if pd.notna(r.deathtime) else np.inf
        end=min(seconds(r.outtime),death);unknown=first_unknown(tl,stable,min(deadline,end))
        state=r.first_drug
        if state=='none':
            state='no_action_unknown_dose' if unknown<min(end,deadline) else ('no_action_observation_ended' if end<deadline else 'no_action_full12h')
        states.append(dict(stay_id=r.stay_id,subject_id=r.subject_id,practice_state=state,stable_t0=r.stable_t0,
            first_action=r.first_drug,action_time=r.action_time,observed_to_12h=min(end,unknown,deadline)-stable))
        if r.first_drug not in [NE,VA]:continue
        action=seconds(r.action_time);A=int(r.first_drug==VA);before=at(tl,action,True);baseline=at(tl,stable,True)
        record=dict(stay_id=r.stay_id,subject_id=r.subject_id,A=A,delay_h=(action-stable)/3600,
            ne_stable=baseline[0],va_stable=baseline[1],ne_action=before[0],va_action=before[1])
        for low,high in WINDOWS:
            lo,hi,v=window(tl,action+low*3600,action+high*3600)
            for left,right,rates in zip(lo,hi,v):
                segments.append(dict(stay_id=r.stay_id,A=A,window_start=low,window_end=high,start_rel_h=(left-action)/3600,
                    end_rel_h=(right-action)/3600,ne=rates[0],va=rates[1]))
            for j,drug in [(0,'ne'),(1,'va')]:
                w=summarize_window(tl,action+low*3600,action+high*3600,j)
                trajectory.append(dict(stay_id=r.stay_id,subject_id=r.subject_id,A=A,window_start=low,window_end=high,drug=drug,**w))
                if low==-2:
                    record[drug+'_net2h']=net_category(w['start_rate'],w['end_rate'],w['full_window'])
                    record[drug+'_relative_change2h']=(w['end_rate']/w['start_rate']-1) if w['full_window'] and w['start_rate']>0 else np.nan
        person.append(record)
    states=pd.DataFrame(states);person=pd.DataFrame(person);trajectory=pd.DataFrame(trajectory)
    states.to_csv(OUT/'stable_pool_practice_states.csv',index=False);person.to_csv(OUT/'decision_practice_individual.csv',index=False)
    trajectory.to_csv(OUT/'pre_action_windows.csv',index=False);pd.DataFrame(segments).to_csv(OUT/'pre_action_exact_segments.csv',index=False)
    counts=states.groupby('practice_state').agg(n_admissions=('stay_id','size'),n_subjects=('subject_id','nunique')).reset_index()
    counts['fraction_pool']=counts.n_admissions/len(pool);counts.to_csv(OUT/'practice_state_counts.csv',index=False)
    net=person.groupby(['A','ne_net2h','va_net2h']).size().rename('n').reset_index()
    net['n_group']=net.A.map(person.A.value_counts());net['fraction_group']=net.n/net.n_group
    net.to_csv(OUT/'two_drug_net_change_patterns.csv',index=False)
    rows=[]
    for (arm,low,high,drug),g in trajectory.groupby(['A','window_start','window_end','drug']):
        full=g[g.full_window.eq(1)]
        rows.append(dict(A=arm,window_start=low,window_end=high,drug=drug,n_group=len(g),n_full=len(full),
            n_partial_or_unknown=int(g.full_window.eq(0).sum()),n_with_any_known=int(g.known_h.gt(0).sum()),
            median_known_h=float(g.known_h.median()),median=float(full['mean'].median()),q25=float(full['mean'].quantile(.25)),q75=float(full['mean'].quantile(.75))))
    pd.DataFrame(rows).to_csv(OUT/'pre_action_window_summary.csv',index=False)
    summary=[]
    for arm,g in person.groupby('A'):
        for col in ['delay_h','ne_stable','va_stable','ne_action','va_action','ne_relative_change2h','va_relative_change2h']:
            vals=g[col].dropna();summary.append(dict(A=arm,variable=col,n=len(vals),median=float(vals.median()),q25=float(vals.quantile(.25)),q75=float(vals.quantile(.75))))
    pd.DataFrame(summary).to_csv(OUT/'decision_practice_summary.csv',index=False)
    prior=pd.read_csv(SRC/'design_feasibility/decision_measurement_diagnostics.csv')
    checks=dict(stable_pool470=len(states)==470,decision284=len(person)==284,
        cohort_counts_reconcile=int(counts.n_admissions.sum())==len(states),pattern_counts_reconcile=int(net.n.sum())==len(person),
        windows_per_decision=bool(trajectory.groupby('stay_id').size().eq(6).all()),
        window_lengths_bounded=bool(trajectory.observed_h.between(0,2).all()&trajectory.known_h.between(0,2).all()),
        no_post_action_segments=bool(pd.DataFrame(segments).end_rel_h.le(0).all()),
        same_decision_cohort=set(person.stay_id)==set(prior.stay_id))
    joined=person.merge(prior[['stay_id','ne','va']],on='stay_id',validate='one_to_one')
    checks['action_doses_match_prior']=bool(np.allclose(joined.ne_action,joined['ne']) and np.allclose(joined.va_action,joined.va))
    summary=dict(status='DESCRIPTIVE_PRACTICE_COMPUTED_PENDING_REPORT',checks=checks,stable_admissions=len(pool),
        stable_subjects=int(pool.subject_id.nunique()),decision_admissions=len(person),decision_subjects=int(person.subject_id.nunique()),
        new_comparative_effects_computed=False,clinical_intent_verified=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');assert all(checks.values()),checks
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
