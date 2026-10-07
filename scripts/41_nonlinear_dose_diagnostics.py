"""Locked four-scenario design diagnostics without reading outcome columns."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

BASE=Path(__file__).resolve().parents[1];SRC=BASE/'rebuild_v3';OUT=SRC/'nonlinear_dose_v1'
spec=importlib.util.spec_from_file_location('overlap38',BASE/'scripts/38_overlap_analysis.py')
ow=importlib.util.module_from_spec(spec);spec.loader.exec_module(ow);core=ow.core
KNOTS={'ne':([0,.02,.05,.1,.3,1.5],.05),'va':([0,.6,1.2,2.4,3.6,12],1.2)}
SCENARIOS=[('all284','linear'),('all284','spline'),('stable272','linear'),('stable272','spline')]

def spline_columns(d):
    extra={}
    for col,(knots,scale) in KNOTS.items():
        raw=d[col].astype(float).replace([np.inf,-np.inf],np.nan)
        raw=raw.fillna(raw.median() if raw.notna().any() else 0)
        if (raw<0).any():raise ValueError('Negative dose cannot enter prescribed log1p basis')
        transformed=np.log1p(raw.to_numpy()/scale).reshape(-1,1)
        tr=SplineTransformer(degree=3,knots=np.log1p(np.asarray(knots)/scale).reshape(-1,1),include_bias=False,extrapolation='linear')
        basis=tr.fit_transform(transformed)
        for j in range(basis.shape[1]):extra[f'{col}_spline_{j}']=basis[:,j]
    return pd.DataFrame(extra,index=d.index)

def weighted_quantile(v,w,p):
    order=np.argsort(v);v=np.asarray(v)[order];w=np.asarray(w)[order]
    return float(v[min(np.searchsorted(np.cumsum(w)/w.sum(),p),len(v)-1)])

def fit(d,model,full=False):
    a=d.A.to_numpy();raw=d[core.COV]
    extended=pd.concat([raw,spline_columns(d)],axis=1) if model=='spline' else raw
    x=core.matrix(extended,list(extended.columns));ps=core.fit_probability(x,a);w=ow.overlap_weights(a,ps)
    if not (np.isfinite(w).all() and (w>0).all()):raise ValueError('Invalid weights')
    original_x=core.matrix(raw,core.COV)
    original_bal=ow.distribution_balance(original_x,a,w,True)
    all_bal=ow.distribution_balance(x,a,w,False)
    dose_bal=original_bal.set_index('covariate')
    rounded=pd.DataFrame({'ne':np.round(d['ne'].to_numpy()/.005)*.005,'va':np.round(d.va.to_numpy()/.1)*.1})
    rounded_bal=ow.distribution_balance(rounded,a,w,True).set_index('covariate')
    di=dict(n=len(d),n_ne=int((a==0).sum()),n_vaso=int((a==1).sum()),model_columns=x.shape[1],
        max_abs_smd=float(original_bal.smd.abs().max()),terms_above_point1=int(original_bal.smd.abs().gt(.1).sum()),
        max_abs_smd_including_basis=float(all_bal.smd.abs().max()),max_original_ks=float(original_bal.ks.max()),
        ps_min=float(ps.min()),ps_max=float(ps.max()))
    for col in ['ne','va']:
        di[col+'_ks']=float(dose_bal.loc[col,'ks']);di[col+'_rounded_ks']=float(rounded_bal.loc[col,'ks'])
        di[col+'_variance_ratio']=float(dose_bal.loc[col,'variance_ratio'])
    for arm,label in [(0,'nefirst'),(1,'vasofirst')]:
        ww=w[a==arm];di['ess_'+label]=float(ww.sum()**2/(ww@ww));di['max_share_'+label]=float(ww.max()/ww.sum())
    if not full:return di
    quant=[]
    for col in ['ne','va','ned','ps']:
        vals=ps if col=='ps' else d[col].to_numpy()
        for arm in [0,1]:
            mask=a==arm;v=vals[mask];ww=w[mask]
            row=dict(variable=col,arm=arm,weighted_mean=float(np.average(v,weights=ww)),
                max_contribution_share=float(np.max(v*ww)/np.sum(v*ww)) if np.sum(v*ww)>0 else np.nan)
            row.update({f'q{int(p*100)}':weighted_quantile(v,ww,p) for p in [.05,.25,.5,.75,.95]});quant.append(row)
    return di,original_bal,all_bal,d[['stay_id','subject_id','A']].assign(ps=ps,overlap_weight=w),pd.DataFrame(quant)

def main():
    OUT.mkdir(exist_ok=True)
    usecols=['stay_id','subject_id','A']+core.COV
    d=pd.read_csv(SRC/'decision_analysis.csv',usecols=usecols)
    flags=pd.read_csv(SRC/'design_feasibility/decision_measurement_diagnostics.csv',usecols=['stay_id','past2h_stable_at_action'])
    d=d.merge(flags,on='stay_id',validate='one_to_one')
    paths=[SRC/'decision_analysis.csv',SRC/'design_feasibility/decision_measurement_diagnostics.csv',BASE/'docs/nonlinear_dose_diagnostic_plan_v1.md',Path(__file__),BASE/'scripts/38_overlap_analysis.py',BASE/'scripts/32_rebuild_inference.py']
    manifest=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    (OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    points=[];boot=[];failed=[]
    with threadpool_limits(limits=1):
        for group,model in SCENARIOS:
            label=group+'_'+model;sub=d if group=='all284' else d[d.past2h_stable_at_action.eq(1)]
            di,bal,ext,w,q=fit(sub,model,True);points.append(dict(scenario=label,**di))
            bal.to_csv(OUT/f'{label}_balance.csv',index=False);ext.to_csv(OUT/f'{label}_basis_balance.csv',index=False)
            w.to_csv(OUT/f'{label}_weights.csv',index=False);q.to_csv(OUT/f'{label}_distribution.csv',index=False)
            if model=='linear':
                prior=pd.read_csv(SRC/f'overlap_v1/{group}_weights.csv');joined=w.merge(prior,on=['stay_id','subject_id','A'],suffixes=('_new','_old'),validate='one_to_one')
                np.testing.assert_allclose(joined.overlap_weight_new,joined.overlap_weight_old,atol=1e-10,rtol=0)
            print('POINT',label,json.dumps(di),flush=True)
        pd.DataFrame(points).to_csv(OUT/'point_diagnostics.csv',index=False)
        rng=np.random.default_rng(2026091502);ids=d.subject_id.unique();groups={s:g for s,g in d.groupby('subject_id',sort=False)}
        for b in range(300):
            sample=pd.concat([groups[s] for s in rng.choice(ids,len(ids),replace=True)],ignore_index=True)
            for group,model in SCENARIOS:
                sub=sample if group=='all284' else sample[sample.past2h_stable_at_action.eq(1)]
                try:boot.append(dict(replicate=b,scenario=group+'_'+model,**fit(sub,model)))
                except (ValueError,FloatingPointError,np.linalg.LinAlgError,ConvergenceWarning) as ex:failed.append(dict(replicate=b,scenario=group+'_'+model,error=str(ex)))
            if (b+1)%50==0:
                print('DIAGNOSTIC BOOTSTRAP',b+1,'/300','successful',len(boot),'failed',len(failed),flush=True)
                pd.DataFrame(boot).to_csv(OUT/'bootstrap_checkpoint.csv',index=False)
        boot=pd.DataFrame(boot);boot.to_csv(OUT/'bootstrap_diagnostics.csv',index=False)
    previous=pd.read_csv(SRC/'overlap_v1/bootstrap_diagnostics.csv')
    for group in ['all284','stable272']:
        new=boot[boot.scenario.eq(group+'_linear')].set_index('replicate');old=previous[previous.scenario.eq(group)].set_index('replicate')
        for n,o in [('max_abs_smd','max_abs_smd'),('ess_nefirst','ess_ne'),('ess_vasofirst','ess_vaso')]:np.testing.assert_allclose(new[n],old.loc[new.index,o],atol=1e-9,rtol=0)
    summary=dict(status='NONLINEAR_DIAGNOSTICS_COMPUTED_PENDING_AUDIT',success_by_scenario=boot.groupby('scenario').size().to_dict(),failures=failed,
        outcome_columns_read=False,read_columns=usecols+['past2h_stable_at_action'],new_effects_computed=False,
        original_linear_point_weights_reproduced=True,all_original_linear_bootstrap_smd_ess_reproduced=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
