"""Paired, fully refitted patient bootstrap across all locked time-grid scenarios."""
from pathlib import Path
import importlib.util
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

BASE=Path(__file__).resolve().parents[1]
SRC=BASE/'rebuild_v3'
OUT=SRC/'weight_time_audit'
spec=importlib.util.spec_from_file_location('inference32',BASE/'scripts/32_rebuild_inference.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
SCENARIOS=[(60,'linear'),(60,'categorical'),(15,'linear'),(15,'categorical')]
B=300

def weighted_hazards(d,w,minutes):
    nb=int(24*60/minutes)
    keep=d.censor.eq(0).to_numpy()&d.obs_censor.eq(0).to_numpy()
    keys=(d.arm.to_numpy()*nb+d.block.to_numpy()).astype(int)
    den=np.bincount(keys[keep],weights=w[keep],minlength=2*nb).reshape(2,nb)
    if (den<=0).any():raise ValueError('Empty weighted risk set')
    squared=np.bincount(keys[keep],weights=w[keep]**2,minlength=2*nb).reshape(2,nb)
    n=np.bincount(keys[keep],minlength=2*nb).reshape(2,nb)
    events=[]
    for col in ['target','compete']:
        events.append(np.bincount(keys[keep],weights=(w*d[col].to_numpy())[keep],minlength=2*nb).reshape(2,nb)/den)
    h1,h2=events
    assert (h1+h2<=1+1e-10).all()
    s=np.cumprod(np.maximum(0,1-h1-h2),axis=1)
    before=np.column_stack([np.ones(2),s[:,:-1]])
    cif=np.cumsum(before*h1,axis=1)
    return cif[:,-1],dict(n=n,ess=den**2/squared,h1=h1,h2=h2,cif=cif)

def fit(d,minutes,time_model,diagnostics=False):
    d=d.sort_values(['clone_id','block']).reset_index(drop=True).copy()
    p=np.ones(len(d));p0=p.copy();nb=int(24*60/minutes);grace=int(12*60/minutes)
    calibration=[]
    for arm in [0,1]:
        mask=d.arm.eq(arm)&d.pending.eq(1)
        sub=d.loc[mask]
        if time_model=='linear':
            x=core.matrix(sub,core.COV+['block'])
        else:
            indicators=pd.get_dummies(pd.Categorical(sub.block,categories=range(grace)),prefix='time',drop_first=True,dtype=float)
            indicators.index=sub.index
            raw=pd.concat([sub[core.COV],indicators],axis=1)
            x=core.matrix(raw,list(raw.columns))
        pr=core.fit_probability(x,1-sub.censor.to_numpy())
        p0[mask]=pr;p[mask]=np.clip(pr,.005,.9995)
        if diagnostics:
            tmp=sub[['block','censor']].copy();tmp['p']=p[mask];tmp['p_raw']=pr
            for k,g in tmp.groupby('block'):
                calibration.append(dict(arm=arm,block=int(k),start_h=k*minutes/60,end_h=(k+1)*minutes/60,
                    n=len(g),n_censored=int(g.censor.sum()),observed_uncensored=float(1-g.censor.mean()),
                    predicted_uncensored=float(g.p.mean()),prediction_raw_mean=float(g.p_raw.mean())))
    raw=1/pd.Series(p).groupby(d.clone_id,sort=False).cumprod().to_numpy()
    trimmed=raw.copy()
    for arm in [0,1]:
        mask=d.arm.eq(arm).to_numpy();trimmed[mask]=np.minimum(raw[mask],np.quantile(raw[mask],.99))
    estimates={};risksets=[]
    keep=d.censor.eq(0)&d.obs_censor.eq(0)
    for name,w in [('trim',trimmed),('raw',raw),('unweighted',np.ones(len(d)))]:
        risk,h=weighted_hazards(d,w,minutes)
        estimates.update({name+'_ne':risk[0],name+'_vaso':risk[1],name+'_rd':risk[1]-risk[0],name+'_rr':risk[1]/risk[0]})
        if diagnostics:
            for arm in [0,1]:
                for k in range(nb):
                    mask=(keep&d.arm.eq(arm)&d.block.eq(k)).to_numpy()
                    risksets.append(dict(weighting=name,arm=arm,block=k,end_h=(k+1)*minutes/60,
                        n=int(h['n'][arm,k]),ess=h['ess'][arm,k],target_hazard=h['h1'][arm,k],
                        compete_hazard=h['h2'][arm,k],cif=h['cif'][arm,k],
                        weight_q50=np.quantile(w[mask],.5),weight_q99=np.quantile(w[mask],.99),weight_max=w[mask].max()))
    if not diagnostics:return estimates
    finite=bool(np.isfinite(raw).all()&np.isfinite(trimmed).all())
    fixed=True
    ww=d[['clone_id','pending']].copy();ww['raw']=raw
    for _,g in ww[ww.pending.eq(0)].groupby('clone_id'):
        fixed=fixed and bool(np.allclose(g.raw,g.raw.iloc[0]))
    diag=dict(minutes=minutes,time_model=time_model,n_rows=len(d),n_outcome_rows=int(keep.sum()),
        raw_weight_max_all=float(raw.max()),raw_weight_max_outcome=float(raw[keep].max()),
        trimmed_weight_max_outcome=float(trimmed[keep].max()),finite_weights=finite,weights_fixed_after_adherence=fixed,
        min_outcome_ess=float(min(r['ess'] for r in risksets if r['weighting']=='trim')),
        clipped_probability_count=int(np.count_nonzero(p!=p0)))
    assert finite and fixed
    return estimates,diag,pd.DataFrame(calibration),pd.DataFrame(risksets)

def sampler(d):
    return {sid:g.index.to_numpy() for sid,g in d.groupby('subject_id',sort=False)}

def resample(d,groups,draw):
    chunks=[groups[s] for s in draw];idx=np.concatenate(chunks)
    q=d.iloc[idx].copy().reset_index(drop=True)
    copies=np.repeat(np.arange(len(draw)),[len(x) for x in chunks])
    q['clone_id']=pd.Series(copies).astype(str)+'_'+q.stay_id.astype(str)+'_'+q.arm.astype(str)
    return q

def main():
    with threadpool_limits(limits=1):
        data={m:pd.read_pickle(OUT/f'grid_{m}.pkl').reset_index(drop=True) for m in [60,15]}
        for d in data.values():d['clone_id']=d.stay_id.astype(str)+'_'+d.arm.astype(str)
        points={};diagnostics=[];cal=[];haz=[]
        for m,tm in SCENARIOS:
            label=f'{m}min_{tm}'
            e,di,c,h=fit(data[m],m,tm,True)
            points[label]=e;di['scenario']=label;diagnostics.append(di)
            c['scenario']=label;h['scenario']=label;cal.append(c);haz.append(h)
            print('POINT',label,json.dumps(e),flush=True)
        old=pd.read_csv(SRC/'ccw_results.csv').set_index('estimand')
        for new,previous in [('trim','ccw'),('raw','ccw_untrimmed'),('unweighted','ccw_unweighted')]:
            for metric in ['ne','vaso','rd','rr']:
                assert np.isclose(points['60min_linear'][new+'_'+metric],old.loc[previous+'_'+metric,'estimate'],atol=1e-10,rtol=0)
        pd.concat(cal).to_csv(OUT/'censor_calibration.csv',index=False)
        pd.concat(haz).to_csv(OUT/'riskset_diagnostics.csv',index=False)
        (OUT/'point_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
        (OUT/'point_estimates.json').write_text(json.dumps(points,indent=2),encoding='utf-8')
        rng=np.random.default_rng(2026091403)
        ids=data[60].subject_id.unique();groups={m:sampler(d) for m,d in data.items()}
        assert set(ids)==set(data[15].subject_id)
        results=[];failures=[]
        for b in range(B):
            draw=rng.choice(ids,len(ids),replace=True)
            samples={m:resample(d,groups[m],draw) for m,d in data.items()}
            for m,tm in SCENARIOS:
                label=f'{m}min_{tm}'
                try:
                    e=fit(samples[m],m,tm);results.append(dict(replicate=b,scenario=label,**e))
                except (ValueError,FloatingPointError,np.linalg.LinAlgError,ConvergenceWarning) as ex:
                    failures.append(dict(replicate=b,scenario=label,error=str(ex)))
            if (b+1)%25==0:
                pd.DataFrame(results).to_csv(OUT/'bootstrap_checkpoint.csv',index=False)
                (OUT/'bootstrap_failures.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
                print('PAIRED BOOTSTRAP',b+1,'/',B,'successful fits',len(results),'failures',len(failures),flush=True)
        boot=pd.DataFrame(results);boot.to_csv(OUT/'bootstrap.csv',index=False)
        rows=[]
        for scenario,point in points.items():
            sub=boot[boot.scenario.eq(scenario)]
            for key,value in point.items():
                vals=sub[key].to_numpy();vals=vals[np.isfinite(vals)]
                lo,hi=np.percentile(vals,[2.5,97.5]) if len(vals) else [np.nan,np.nan]
                rows.append(dict(scenario=scenario,estimand=key,estimate=value,lo=lo,hi=hi,bootstrap_finite=len(vals)))
        pd.DataFrame(rows).to_csv(OUT/'sensitivity_results.csv',index=False)
        pivot=boot.pivot(index='replicate',columns='scenario',values='trim_rd');paired=[]
        for label in points:
            if label=='60min_linear':continue
            vals=(pivot[label]-pivot['60min_linear']).dropna()
            lo,hi=np.percentile(vals,[2.5,97.5]) if len(vals) else [np.nan,np.nan]
            paired.append(dict(scenario=label,reference='60min_linear',estimand='paired_difference_in_rd',
                estimate=points[label]['trim_rd']-points['60min_linear']['trim_rd'],lo=lo,hi=hi,n_pairs=len(vals)))
        pd.DataFrame(paired).to_csv(OUT/'paired_scenario_differences.csv',index=False)
        # Same seed and patient order: verify all baseline bootstrap estimates, not only the point.
        prior=pd.read_csv(SRC/'ccw_bootstrap.csv').set_index('replicate')
        baseline=boot[boot.scenario.eq('60min_linear')].set_index('replicate')
        reproduced=bool(np.allclose(prior.loc[baseline.index,'ccw_rd'],baseline.trim_rd,rtol=0,atol=1e-9))
        summary=dict(status='SENSITIVITY_COMPUTED_PENDING_REPORT_AUDIT',requested_per_scenario=B,
            success_by_scenario=boot.groupby('scenario').size().to_dict(),failures=failures,
            baseline_point_reproduced=True,baseline_bootstrap_reproduced=reproduced,causal_assumptions_verified=False)
        (OUT/'inference_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
