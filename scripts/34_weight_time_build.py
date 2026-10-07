"""Build finer CCW grid without changing exact endpoints; descriptive support audit."""
from pathlib import Path
import importlib.util
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
from rebuild_v3_core import seconds, VA, NE

BASE=Path(__file__).resolve().parents[1]
SRC=BASE/'rebuild_v3'
OUT=SRC/'weight_time_audit'

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,BASE/'scripts'/file)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

ds=module('datasets31','31_rebuild_datasets.py')
inf=module('inference32','32_rebuild_inference.py')

def grid_rows(stop,terminal,action,own,minutes):
    """End-boundary events belong to the completing bin, including grace deadline."""
    step=minutes/60
    for k in range(int(24/step)):
        a=k*step;z=(k+1)*step
        if stop<a or (stop==a and k>0):break
        done=own and (action<=a if k else action<a)
        yield dict(block=k,pending=int(not done and a<12),censor=int(terminal==3 and stop<=z),
            target=int(terminal==1 and stop<=z),compete=int(terminal==2 and stop<=z),
            obs_censor=int(terminal==0 and stop<z),stop_h=stop)
        if stop<=z:break

def main():
    OUT.mkdir(exist_ok=True)
    inputs=[SRC/x for x in ['stable_pool.csv','ccw_clones.csv','ccw_long.csv','decision_analysis.csv','decision_weights.csv','timelines.pkl']]
    inputs+=[BASE/'docs/weight_time_audit_plan_v1.md',Path(__file__)]
    inputs+=list(SRC.glob('raw_*.pkl'))
    manifest=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs]
    (OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    d=pd.read_csv(SRC/'decision_analysis.csv').merge(pd.read_csv(SRC/'decision_weights.csv').rename(columns={'weight':'iptw'}),on=['stay_id','A'],validate='one_to_one')
    x=inf.matrix(d,ds.BASE_COV);a=d.A.to_numpy()
    bal=[]
    for col in x:
        v=x[col].to_numpy();sd=np.sqrt((np.var(v[a==0])+np.var(v[a==1]))/2)
        for name,w in [('unweighted',np.ones(len(d))),('iptw',d.iptw.to_numpy()),('untrimmed',d.weight_untrimmed.to_numpy())]:
            diff=np.average(v[a==1],weights=w[a==1])-np.average(v[a==0],weights=w[a==0])
            bal.append(dict(covariate=col,weighting=name,smd_fixed_baseline_sd=diff/sd if sd>0 else 0,sd=sd))
    pd.DataFrame(bal).to_csv(OUT/'decision_balance_fixed_sd.csv',index=False)
    desc=[]
    for arm,g in d.groupby('A'):
        for col in ['ps','ne','va','ned','iptw','weight_untrimmed']:
            desc.append(dict(arm=int(arm),variable=col,n=len(g),**{f'q{int(q*100)}':float(g[col].quantile(q)) for q in [0,.01,.05,.25,.5,.75,.95,.99,1]}))
    pd.DataFrame(desc).to_csv(OUT/'decision_support_distribution.csv',index=False)
    support=[]
    common_lo=max(d.loc[d.A.eq(g),'ps'].min() for g in [0,1])
    common_hi=min(d.loc[d.A.eq(g),'ps'].max() for g in [0,1])
    for label,lo,hi in [('empirical_common_range',common_lo,common_hi),('ps_05_95',.05,.95),('ps_10_90',.1,.9)]:
        for arm,g in d.groupby('A'):
            support.append(dict(rule=label,arm=int(arm),lower=lo,upper=hi,n_total=len(g),n_in=int(g.ps.between(lo,hi).sum())))
    pd.DataFrame(support).to_csv(OUT/'decision_support_counts.csv',index=False)
    pool=pd.read_csv(SRC/'stable_pool.csv')
    for col in ['stable_t0','intime','surgery_date']:
        pool[col]=pd.to_datetime(pool[col])
    patients={r.stay_id:r for r in pool.itertuples()}
    clones=pd.read_csv(SRC/'ccw_clones.csv')
    old=pd.read_csv(SRC/'ccw_long.csv')
    # Confirm the isolated grid generator exactly reproduces the existing 60-min rows.
    check=[]
    for c in clones.itertuples():
        r=patients[c.stay_id];own=r.first_drug==(VA if c.arm else NE)
        for row in grid_rows(c.stop_h,c.terminal,c.action_h,own,60):
            check.append(dict(stay_id=c.stay_id,arm=c.arm,**row))
    compare_cols=['stay_id','arm','block','pending','censor','target','compete','obs_censor','stop_h']
    expected=old[compare_cols].sort_values(compare_cols[:3]).reset_index(drop=True)
    actual=pd.DataFrame(check)[compare_cols].sort_values(compare_cols[:3]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,atol=1e-10,rtol=0)
    old.to_pickle(OUT/'grid_60.pkl')
    timelines=pickle.load((SRC/'timelines.pkl').open('rb'))
    reader=ds.FeatureReader();fine=[]
    for n,(sid,cc) in enumerate(clones.groupby('stay_id',sort=False),1):
        r=patients[sid];tl=timelines[sid];cache={}
        for c in cc.itertuples():
            own=r.first_drug==(VA if c.arm else NE)
            for row in grid_rows(c.stop_h,c.terminal,c.action_h,own,15):
                k=row['block']
                # Covariates after adherence are unused by the censoring fit.
                if row['pending']:
                    if k not in cache:cache[k]=reader.features(r,tl,seconds(r.stable_t0)+k*900)
                    feat=cache[k]
                else:feat={col:np.nan for col in ds.BASE_COV}
                fine.append(dict(stay_id=sid,subject_id=c.subject_id,arm=c.arm,**row,**feat))
        if n%50==0:print('fine grid',n,'/',len(pool),flush=True)
    fine=pd.DataFrame(fine)
    keys=['stay_id','arm']
    terminals=['target','compete','censor','obs_censor']
    # Unknown censoring exactly at a bin boundary need not have a partial-bin flag.
    for col in terminals[:3]:
        pd.testing.assert_series_equal(old.groupby(keys)[col].sum(),fine.groupby(keys)[col].sum(),check_dtype=False)
    assert not fine.duplicated(keys+['block']).any()
    assert (fine.groupby(keys)[terminals].sum().sum(axis=1)<=1).all()
    assert len(fine[(fine.censor==1)&(fine.stop_h==12)&(fine.block==47)])==len(old[(old.censor==1)&(old.stop_h==12)&(old.block==11)])
    fine.to_pickle(OUT/'grid_15.pkl')
    summary=dict(status='FINE_GRID_BUILT_PENDING_SENSITIVITY',hourly_grid_reproduced=True,
        exact_clinical_and_artificial_terminal_counts_match=True,rows_60=len(old),rows_15=len(fine),
        source_time_feature_checks=reader.source_assertions,cohort_n=len(pool))
    (OUT/'build_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
