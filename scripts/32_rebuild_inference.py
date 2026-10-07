"""Core MIMIC reanalysis with per-patient bootstrap and full nuisance refits."""
from pathlib import Path
import importlib.util
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from rebuild_v3_core import weighted_cif

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'rebuild_v3'
spec=importlib.util.spec_from_file_location('datasets31',BASE/'scripts/31_rebuild_datasets.py')
ds=importlib.util.module_from_spec(spec);spec.loader.exec_module(ds)
COV=ds.BASE_COV
B=300
CONTINUOUS=['dned_peak_6h','dned_peak_12h','dned_peak_24h','dned_mean_6h',
    'ned_mean_0_6','ned_mean_6_12','ned_mean_12_24',
    'map_0_6','map_6_12','map_12_24','hypo65_0_6','lactate_0_6','lactate_6_12','lactate_12_24']


def matrix(d,cols):
    raw=d[cols].astype(float).replace([np.inf,-np.inf],np.nan)
    # Fixed missing indicators for all prespecified columns, even when all observed.
    x=raw.fillna(raw.median().fillna(0))
    x=pd.concat([x,raw.isna().astype(float).add_suffix('_missing')],axis=1)
    return ((x-x.mean())/x.std(ddof=0).replace(0,1)).fillna(0)


def fit_probability(x,y):
    if len(np.unique(y))<2:
        raise ValueError('One outcome class in nuisance fit')
    m=LogisticRegression(C=1.,max_iter=3000)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        m.fit(x,y)
    return m.predict_proba(x)[:,1]


def smd(x,a,w):
    x=np.asarray(x,float);a=np.asarray(a);w=np.asarray(w)
    m=[np.average(x[a==g],weights=w[a==g]) for g in [0,1]]
    v=[np.average((x[a==g]-m[g])**2,weights=w[a==g]) for g in [0,1]]
    sd=np.sqrt(sum(v)/2)
    return (m[1]-m[0])/sd if sd>0 else 0.


def decision_fit(d,return_diag=False):
    a=d.A.to_numpy();x=matrix(d,COV);ps=fit_probability(x,a)
    prob=a.mean();raw=np.where(a==1,prob/ps,(1-prob)/(1-ps))
    w=np.clip(raw,*np.quantile(raw,[.01,.99]))
    effects={}
    for h in [6,12,24]:
        for weighting,ww in [('iptw',w),('unweighted',np.ones(len(d))),('untrimmed',raw)]:
            risk=[weighted_cif(d.loc[a==g,'time_h'],d.loc[a==g,'event'],ww[a==g],h) for g in [0,1]]
            prefix=f'{weighting}_relapse{h}'
            effects[prefix+'_ne']=risk[0];effects[prefix+'_vaso']=risk[1]
            effects[prefix+'_rd']=risk[1]-risk[0]
            effects[prefix+'_rr']=risk[1]/risk[0] if risk[0]>0 else np.nan
    for col in CONTINUOUS:
        y=d[col].to_numpy(float);mask=np.isfinite(y)
        means=[np.average(y[mask&(a==g)],weights=w[mask&(a==g)]) if (mask&(a==g)).any() else np.nan for g in [0,1]]
        effects[col+'_ne']=means[0];effects[col+'_vaso']=means[1];effects[col+'_diff']=means[1]-means[0]
    full=d.full_6h.eq(1).to_numpy();y=d.dned_peak_6h.to_numpy(float)
    means=[np.average(y[full&(a==g)],weights=w[full&(a==g)]) if (full&(a==g)).any() else np.nan for g in [0,1]]
    effects['dned_peak_6h_fullwindow_diff']=means[1]-means[0]
    if not return_diag:return effects
    balance=pd.DataFrame([dict(covariate=col,smd_unweighted=smd(x[col],a,np.ones(len(d))),smd_weighted=smd(x[col],a,w)) for col in x])
    diag=dict(n=len(d),n_ne=int((a==0).sum()),n_vaso=int((a==1).sum()),
        ps_min=float(ps.min()),ps_max=float(ps.max()),weight_raw_max=float(raw.max()),weight_trimmed_max=float(w.max()),
        ess=float(w.sum()**2/(w@w)),ess_ne=float(w[a==0].sum()**2/(w[a==0]@w[a==0])),ess_vaso=float(w[a==1].sum()**2/(w[a==1]@w[a==1])),
        max_smd_before=float(balance.smd_unweighted.abs().max()),max_smd_after=float(balance.smd_weighted.abs().max()),
        covariates_above_0_1=int((balance.smd_weighted.abs()>.1).sum()))
    return effects,diag,balance,pd.DataFrame({'stay_id':d.stay_id,'A':a,'ps':ps,'weight':w,'weight_untrimmed':raw})


def ccw_fit(long,return_diag=False):
    d=long.copy().sort_values(['clone_id','block']).reset_index(drop=True)
    d['p_uncensored']=1.
    diagnostics=[]
    for arm in [0,1]:
        mask=d.arm.eq(arm)&d.pending.eq(1)
        sub=d.loc[mask]
        x=matrix(sub,COV+['block'])
        y=1-sub.censor.to_numpy()
        p=fit_probability(x,y)
        d.loc[mask,'p_uncensored']=np.clip(p,.005,.9995)
        diagnostics.append(dict(arm=arm,n_censor_training=len(sub),n_censored=int(sub.censor.sum()),
            min_p=float(p.min()),max_p=float(p.max())))
    d['weight_raw']=1/d.groupby('clone_id',sort=False).p_uncensored.cumprod()
    # Never mix arm distributions when selecting truncation bounds.
    d['weight']=d.weight_raw
    for arm in [0,1]:
        mask=d.arm.eq(arm)
        d.loc[mask,'weight']=d.loc[mask,'weight_raw'].clip(upper=d.loc[mask,'weight_raw'].quantile(.99))
    keep=(d.censor==0)&(d.obs_censor==0)
    estimates={};hazards=[]
    for name,wcol in [('ccw','weight'),('ccw_untrimmed','weight_raw'),('ccw_unweighted',None)]:
        risks={}
        for arm in [0,1]:
            survival,cif=1.,0.
            for k in range(24):
                sub=d[keep&d.arm.eq(arm)&d.block.eq(k)]
                if not len(sub):
                    raise ValueError(f'Empty CCW risk set arm={arm} hour={k}')
                w=sub[wcol].to_numpy() if wcol else np.ones(len(sub))
                h1=float(w@sub.target.to_numpy()/w.sum());h2=float(w@sub.compete.to_numpy()/w.sum())
                assert h1+h2<=1+1e-10
                cif+=survival*h1;survival*=max(0,1-h1-h2)
                if name=='ccw':hazards.append(dict(arm=arm,block=k,n=len(sub),target_hazard=h1,compete_hazard=h2,cif=cif,
                    ess=float(w.sum()**2/(w@w)),weight_max=float(w.max())))
            risks[arm]=cif
        estimates[name+'_ne']=risks[0];estimates[name+'_vaso']=risks[1]
        estimates[name+'_rd']=risks[1]-risks[0];estimates[name+'_rr']=risks[1]/risks[0] if risks[0]>0 else np.nan
    if not return_diag:return estimates
    diag=dict(n_patients=int(d.subject_id.nunique()),n_stays=int(d.stay_id.nunique()),n_clones=int(d.clone_id.nunique()),
        n_rows=len(d),n_outcome_rows=int(keep.sum()),weight_raw_max=float(d.weight_raw.max()),weight_max=float(d.weight.max()),
        min_block_ess=float(min(x['ess'] for x in hazards)),
        censor_models=diagnostics,method='hourly inverse artificial-censoring weighting; ICU discharge competing')
    return estimates,diag,pd.DataFrame(hazards),d[['stay_id','subject_id','clone_id','arm','block','pending','censor','target','compete','obs_censor','p_uncensored','weight_raw','weight']]


def bootstrap(data,fit,ccw=False):
    rng=np.random.default_rng(2026091403 if ccw else 2026091402)
    ids=data.subject_id.unique();groups={s:g for s,g in data.groupby('subject_id',sort=False)}
    results=[];failures=[]
    for b in range(B):
        parts=[]
        for j,s in enumerate(rng.choice(ids,len(ids),replace=True)):
            q=groups[s].copy()
            if ccw:q['clone_id']=str(j)+'_'+q.stay_id.astype(str)+'_'+q.arm.astype(str)
            parts.append(q)
        sample=pd.concat(parts,ignore_index=True)
        try:
            e=fit(sample);e['replicate']=b;results.append(e)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError,ConvergenceWarning) as ex:
            failures.append(dict(replicate=b,error=str(ex)))
        if (b+1)%50==0:print('CCW' if ccw else 'decision','bootstrap',b+1,'success',len(results),flush=True)
    return pd.DataFrame(results),failures


def summarize(points,boot):
    rows=[]
    for key,value in points.items():
        vals=boot[key].dropna().to_numpy()
        ci=np.percentile(vals,[2.5,97.5]) if len(vals) else [np.nan,np.nan]
        rows.append(dict(estimand=key,estimate=value,lo=ci[0],hi=ci[1],bootstrap_finite=len(vals)))
    return pd.DataFrame(rows)


def main():
    with threadpool_limits(limits=1):
        d=pd.read_csv(OUT/'decision_analysis.csv')
        long=pd.read_csv(OUT/'ccw_long.csv')
        long['clone_id']=long.stay_id.astype(str)+'_'+long.arm.astype(str)
        p,dd,bal,ww=decision_fit(d,True)
        bal.to_csv(OUT/'decision_balance.csv',index=False);ww.to_csv(OUT/'decision_weights.csv',index=False)
        db,df=bootstrap(d,decision_fit)
        db.to_csv(OUT/'decision_bootstrap.csv',index=False)
        summarize(p,db).to_csv(OUT/'decision_results.csv',index=False)
        print('DECISION',json.dumps(dd),flush=True)
        p2,cd,haz,cw=ccw_fit(long,True)
        haz.to_csv(OUT/'ccw_hazards.csv',index=False);cw.to_csv(OUT/'ccw_weights.csv',index=False)
        cb,cf=bootstrap(long,ccw_fit,True)
        cb.to_csv(OUT/'ccw_bootstrap.csv',index=False)
        summarize(p2,cb).to_csv(OUT/'ccw_results.csv',index=False)
        report=dict(decision=dd,ccw=cd,bootstrap_requested=B,decision_bootstrap_success=len(db),ccw_bootstrap_success=len(cb),
            decision_failures=df,ccw_failures=cf,
            status='CORE_ESTIMATES_COMPUTED_PENDING_AUDIT',causal_assumptions_verified=False)
        (OUT/'inference_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
