"""Post hoc decision-time overlap association, paired full-refit bootstrap."""
from pathlib import Path
import importlib.util
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from rebuild_v3_core import weighted_cif

BASE=Path(__file__).resolve().parents[1];SRC=BASE/'rebuild_v3';OUT=SRC/'overlap_v1'
spec=importlib.util.spec_from_file_location('inf32',BASE/'scripts/32_rebuild_inference.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
B=300
CONT=['dned_peak_6h','map_0_6','hypo65_0_6','lactate_0_6','lactate_6_12','lactate_12_24']
PARTS=['restart','escalation','new_drug','concurrent_escalation_or_new_drug']

def overlap_weights(a,ps):return np.where(np.asarray(a)==1,1-np.asarray(ps),np.asarray(ps))

def component_code(event,reason,part):
    event=np.asarray(event);reason=np.asarray(reason)
    return np.where((event==1)&(reason==part),1,np.where(event>0,2,0))

def distribution_balance(x,a,w,full=False):
    rows=[]
    for col in x:
        v=x[col].to_numpy();means=[];var=[]
        sd=np.sqrt((np.var(v[a==0])+np.var(v[a==1]))/2)
        for g in [0,1]:
            m=np.average(v[a==g],weights=w[a==g]);means.append(m)
            var.append(np.average((v[a==g]-m)**2,weights=w[a==g]))
        r=dict(covariate=col,smd=(means[1]-means[0])/sd if sd else 0.)
        if full:
            support=np.unique(v);cdfs=[]
            for g in [0,1]:
                vals=v[a==g];ww=w[a==g];order=np.argsort(vals);vals=vals[order];ww=ww[order]
                cum=np.r_[0,np.cumsum(ww)/ww.sum()]
                cdfs.append(cum[np.searchsorted(vals,support,side='right')])
            r.update(ks=float(np.max(np.abs(cdfs[1]-cdfs[0]))),variance_ratio=var[1]/var[0] if var[0]>0 else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)

def fit(d,full=False):
    a=d.A.to_numpy();x=core.matrix(d,core.COV);ps=core.fit_probability(x,a);w=overlap_weights(a,ps)
    if not (np.isfinite(w).all() and (w>0).all()):raise ValueError('Nonpositive/nonfinite overlap weights')
    effects={};coverage=[]
    for h in [6,12,24]:
        risks=[weighted_cif(d.loc[a==g,'time_h'],d.loc[a==g,'event'],w[a==g],h) for g in [0,1]]
        effects.update({f'relapse{h}_ne':risks[0],f'relapse{h}_vaso':risks[1],f'relapse{h}_rd':risks[1]-risks[0],f'relapse{h}_rr':risks[1]/risks[0] if risks[0]>0 else np.nan})
    sums=np.zeros(2)
    for part in PARTS:
        code=component_code(d.event,d.event_reason,part)
        risks=[weighted_cif(d.loc[a==g,'time_h'],code[a==g],w[a==g],24) for g in [0,1]]
        sums+=risks
        effects.update({part+'_ne':risks[0],part+'_vaso':risks[1],part+'_rd':risks[1]-risks[0]})
    np.testing.assert_allclose(sums,[effects['relapse24_ne'],effects['relapse24_vaso']],atol=1e-10,rtol=0)
    for col in CONT:
        y=d[col].to_numpy();m=[]
        for g in [0,1]:
            mask=(a==g)&np.isfinite(y);ww=w[mask]
            m.append(float(np.average(y[mask],weights=ww)) if len(ww) else np.nan)
            if full:coverage.append(dict(outcome=col,arm=g,n_group=int((a==g).sum()),n_observed=int(mask.sum()),ess_observed=float(ww.sum()**2/(ww@ww)) if len(ww) else 0))
        effects.update({col+'_ne':m[0],col+'_vaso':m[1],col+'_diff':m[1]-m[0]})
    bal=distribution_balance(x,a,w,full)
    diag=dict(n=len(d),patients=int(d.subject_id.nunique()),n_ne=int((a==0).sum()),n_vaso=int((a==1).sum()),
        max_abs_smd=float(bal.smd.abs().max()),terms_above_point1=int(bal.smd.abs().gt(.1).sum()),ps_min=float(ps.min()),ps_max=float(ps.max()))
    for g,label in [(0,'ne'),(1,'vaso')]:
        ww=w[a==g];diag['ess_'+label]=float(ww.sum()**2/(ww@ww));diag['max_share_'+label]=float(ww.max()/ww.sum())
    if not full:return effects,diag
    weight_table=d[['stay_id','subject_id','A']].assign(ps=ps,overlap_weight=w)
    return effects,diag,bal,weight_table,pd.DataFrame(coverage)

def main():
    OUT.mkdir(exist_ok=True)
    inputs=[SRC/'decision_analysis.csv',SRC/'design_feasibility/decision_measurement_diagnostics.csv',SRC/'design_feasibility/overlap_weights_diagnostic_only.csv',BASE/'docs/overlap_analysis_amendment_v1.md',Path(__file__),BASE/'scripts/32_rebuild_inference.py',BASE/'scripts/rebuild_v3_core.py']
    (OUT/'input_manifest.json').write_text(json.dumps([dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs],indent=2),encoding='utf-8')
    d=pd.read_csv(SRC/'decision_analysis.csv')
    flags=pd.read_csv(SRC/'design_feasibility/decision_measurement_diagnostics.csv')[['stay_id','past2h_stable_at_action']]
    d=d.merge(flags,on='stay_id',validate='one_to_one')
    scenarios={'all284':lambda z:z,'stable272':lambda z:z[z.past2h_stable_at_action.eq(1)]}
    points={};diags=[]
    with threadpool_limits(limits=1):
        for name,subset in scenarios.items():
            e,di,bal,w,cov=fit(subset(d),True);points[name]=e;diags.append(dict(scenario=name,**di))
            bal.to_csv(OUT/f'{name}_balance.csv',index=False);w.to_csv(OUT/f'{name}_weights.csv',index=False);cov.to_csv(OUT/f'{name}_coverage.csv',index=False)
            if name=='all284':
                old=pd.read_csv(SRC/'design_feasibility/overlap_weights_diagnostic_only.csv')
                check=w.merge(old,on=['stay_id','subject_id','A'],suffixes=('_new','_old'),validate='one_to_one')
                np.testing.assert_allclose(check.overlap_weight_new,check.overlap_weight_old,atol=1e-10,rtol=0)
            print('POINT',name,json.dumps({k:v for k,v in e.items() if k.startswith('relapse24')}),flush=True)
        pd.DataFrame(diags).to_csv(OUT/'point_diagnostics.csv',index=False)
        rng=np.random.default_rng(2026091502);ids=d.subject_id.unique();groups={s:g for s,g in d.groupby('subject_id',sort=False)}
        boot=[];bd=[];failures=[]
        for b in range(B):
            sample=pd.concat([groups[s] for s in rng.choice(ids,len(ids),replace=True)],ignore_index=True)
            for name,subset in scenarios.items():
                try:
                    e,di=fit(subset(sample));boot.append(dict(replicate=b,scenario=name,**e));bd.append(dict(replicate=b,scenario=name,**di))
                except (ValueError,FloatingPointError,np.linalg.LinAlgError,ConvergenceWarning) as ex:
                    failures.append(dict(replicate=b,scenario=name,error=str(ex)))
            if (b+1)%50==0:print('BOOTSTRAP',b+1,'/300','successful fits',len(boot),'failures',len(failures),flush=True)
    boot=pd.DataFrame(boot);bd=pd.DataFrame(bd)
    boot.to_csv(OUT/'bootstrap.csv',index=False);bd.to_csv(OUT/'bootstrap_diagnostics.csv',index=False)
    rows=[]
    for name,p in points.items():
        for key,val in p.items():
            values=boot.loc[boot.scenario.eq(name),key].to_numpy();values=values[np.isfinite(values)]
            lo,hi=np.percentile(values,[2.5,97.5]) if len(values) else [np.nan,np.nan]
            rows.append(dict(scenario=name,estimand=key,estimate=val,lo=lo,hi=hi,bootstrap_finite=len(values)))
    pd.DataFrame(rows).to_csv(OUT/'results.csv',index=False)
    status=dict(status='OVERLAP_COMPUTED_PENDING_AUDIT',bootstrap_requested=300,success_by_scenario=boot.groupby('scenario').size().to_dict(),failures=failures,
        original_diagnostic_weights_reproduced=True,component_cif_identity_all_fits_pass=True,causal_assumptions_verified=False,submission_ready=False)
    (OUT/'inference_summary.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2),flush=True)

if __name__=='__main__':main()
