"""Outcome-effect-free design diagnostics; no cohort or effect estimate mutation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import pickle
import subprocess
import sys
import numpy as np
import pandas as pd
from rebuild_v3_core import NE,VA,seconds,window,at,first_unknown

BASE=Path(__file__).resolve().parents[1]
SRC=BASE/'rebuild_v3'
OUT=SRC/'design_feasibility'
spec=importlib.util.spec_from_file_location('inf32',BASE/'scripts/32_rebuild_inference.py')
inf=importlib.util.module_from_spec(spec);spec.loader.exec_module(inf)

def past_stable(tl,t):
    lo,hi,v=window(tl,t-7200,t)
    good=np.isfinite(v[:,:2]).all(axis=1)&(v[:,:2]>0).all(axis=1)&(v[:,0]<=.1001)&(v[:,1]<=3.003)
    return bool(abs((hi-lo).sum()-7200)<1e-6 and good.all())

def restart_observation(tl,action,j,end,minutes):
    horizon=action+minutes*60
    z=min(end,horizon)
    lo,hi,v=window(tl,action,z)
    positive=v[:,j]>0
    if positive.any():return 1,float((lo[positive][0]-action)/60)
    # A restart exactly at the horizon is counted if observation extends past it.
    if end>horizon and at(tl,horizon)[j]>0:return 1,float(minutes)
    return (0 if end>=horizon else None),None

def main():
    OUT.mkdir(exist_ok=True)
    paths=[SRC/x for x in ['stable_pool.csv','decision_analysis.csv','decision_weights.csv','timelines.pkl','source_pressors.pkl']]
    paths += [BASE/'docs/design_feasibility_plan_v1.md',Path(__file__)]
    manifests=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    (OUT/'input_manifest.json').write_text(json.dumps(manifests,indent=2),encoding='utf-8')
    pool=pd.read_csv(SRC/'stable_pool.csv')
    for col in ['stable_t0','action_time','outtime','deathtime']:pool[col]=pd.to_datetime(pool[col])
    pool['action_h']=(pool.action_time-pool.stable_t0).dt.total_seconds()/3600
    timings=[]
    for h in [1,2,4,6,12]:
        chosen=pool.action_h.le(h)
        timings.append(dict(hour=h,n_pool=len(pool),ne=int((chosen&pool.first_drug.eq(NE)).sum()),
            vaso=int((chosen&pool.first_drug.eq(VA)).sum()),simultaneous=int((chosen&pool.first_drug.eq('simultaneous')).sum()),
            no_recorded_action_yet=int((~chosen).sum())))
    pd.DataFrame(timings).to_csv(OUT/'first_action_timing.csv',index=False)
    d=pd.read_csv(SRC/'decision_analysis.csv',usecols=['stay_id','subject_id','A','t0']+inf.COV)
    weights=pd.read_csv(SRC/'decision_weights.csv').rename(columns={'weight':'iptw'})
    d=d.merge(weights,on=['stay_id','A'],validate='one_to_one')
    x=inf.matrix(d,inf.COV);a=d.A.to_numpy();ow=np.where(a==1,1-d.ps,d.ps)
    balances=[];ess=[]
    for label,w in [('unweighted',np.ones(len(d))),('iptw',d.iptw.to_numpy()),('overlap_diagnostic',ow)]:
        for col in x:
            v=x[col].to_numpy();sd=np.sqrt((np.var(v[a==0])+np.var(v[a==1]))/2)
            diff=np.average(v[a==1],weights=w[a==1])-np.average(v[a==0],weights=w[a==0])
            balances.append(dict(weighting=label,covariate=col,smd=diff/sd if sd>0 else 0))
        for arm in [0,1]:
            ww=w[a==arm]
            ess.append(dict(weighting=label,arm=arm,n=int((a==arm).sum()),ess=float(ww.sum()**2/(ww@ww)),
                largest_normalized_weight=float(ww.max()/ww.sum()),total_weight=float(ww.sum())))
    balances=pd.DataFrame(balances);balances.to_csv(OUT/'balance.csv',index=False)
    pd.DataFrame(ess).to_csv(OUT/'effective_sample_sizes.csv',index=False)
    d[['stay_id','subject_id','A','ps']].assign(overlap_weight=ow).to_csv(OUT/'overlap_weights_diagnostic_only.csv',index=False)
    timelines=pickle.load((SRC/'timelines.pkl').open('rb'))
    bystay={r.stay_id:r for r in pool.itertuples()};rows=[]
    for r in d.itertuples():
        p=bystay[r.stay_id];tl=timelines[r.stay_id];action=seconds(p.action_time)
        death=seconds(p.deathtime) if pd.notna(p.deathtime) else np.inf
        end=min(death,seconds(p.outtime));unknown=first_unknown(tl,action,min(action+7200,end));end=min(end,unknown)
        rec=dict(stay_id=r.stay_id,A=r.A,action_h=p.action_h,ne=r.ne,va=r.va,
            past2h_stable_at_action=int(past_stable(tl,action)),
            positive_low_dose_at_action=int(0<r.ne<=.1001 and 0<r.va<=3.003))
        j=1 if r.A else 0
        for minutes in [5,15,30,120]:
            flag,gap=restart_observation(tl,action,j,end,minutes)
            rec[f'restart_within_{minutes}min']=flag
            rec[f'first_restart_minutes_within_{minutes}']=gap
        assert np.isclose(at(tl,action,True)[0],r.ne) and np.isclose(at(tl,action,True)[1],r.va)
        rows.append(rec)
    details=pd.DataFrame(rows);details.to_csv(OUT/'decision_measurement_diagnostics.csv',index=False)
    # Fixed, auditable source-record review set; no adjudication is assumed.
    selected={}
    for r in details.itertuples():
        reasons=[]
        if r.restart_within_15min==1:reasons.append('restart_within15min')
        if not r.past2h_stable_at_action:reasons.append('not_past2h_stable')
        if reasons:selected[r.stay_id]=reasons
    rng=np.random.default_rng(2026091501)
    for arm in [0,1]:
        candidates=details[(details.A==arm)&~details.stay_id.isin(selected)].stay_id.to_numpy()
        for sid in rng.choice(candidates,min(10,len(candidates)),replace=False):selected[int(sid)]=['random_comparison']
    traces=[]
    for sid,reasons in selected.items():
        p=bystay[sid];action=seconds(p.action_time);tl=timelines[sid]
        lo,hi,v=window(tl,action-7200,action+7200)
        for left,right,rates in zip(lo,hi,v):
            traces.append(dict(stay_id=sid,review_reason=';'.join(reasons),start_min=(left-action)/60,
                end_min=(right-action)/60,ne=rates[0],va=rates[1],epi=rates[2],phe=rates[3],dopa=rates[4],angii=rates[5]))
    pd.DataFrame(traces).to_csv(OUT/'review_timeline_segments.csv',index=False)
    raw=pd.read_pickle(SRC/'source_pressors.pkl')
    source=[]
    for sid,reasons in selected.items():
        p=bystay[sid];t=p.action_time
        q=raw[(raw.stay_id==sid)&(pd.to_datetime(raw.starttime)<t+pd.Timedelta(hours=2))&(pd.to_datetime(raw.endtime)>t-pd.Timedelta(hours=2))].copy()
        q['review_reason']=';'.join(reasons);source.append(q)
    pd.concat(source,ignore_index=True).to_csv(OUT/'review_source_infusion_rows.csv',index=False)
    evidence=[]
    for sid,reasons in selected.items():
        p=bystay[sid];t=p.action_time;action=seconds(t);tl=timelines[sid]
        q=raw[(raw.stay_id==sid)&raw.drug.eq(p.first_drug)].copy()
        q['starttime']=pd.to_datetime(q.starttime);q['endtime']=pd.to_datetime(q.endtime)
        ending=q[q.endtime.eq(t)&q.starttime.lt(t)&q.rate.gt(0)&q.statusdescription.isin(['ChangeDose/Rate','Changed','FinishedRunning','Stopped','Paused'])].sort_values(['starttime','rate'])
        assert len(ending)>0,f'No source row for reconstructed cessation {sid}'
        endrow=ending.iloc[-1]
        later=q[q.starttime.gt(t)&q.starttime.le(t+pd.Timedelta(minutes=15))&q.rate.gt(0)&q.statusdescription.isin(['ChangeDose/Rate','Changed','FinishedRunning','Stopped','Paused'])].sort_values(['starttime','rate'])
        lo,hi,v=window(tl,action-7200,action)
        bad=(v[:,0]>.1001)|(v[:,1]>3.003)|~np.isfinite(v[:,:2]).all(axis=1)|(v[:,:2]<=0).any(axis=1)
        e=dict(stay_id=sid,reason=';'.join(reasons),drug=p.first_drug,cessation_time=str(t),
            source_row_end_status=endrow.statusdescription,source_end_matches_action=True,
            end_rate_raw=float(endrow.rate),rateuom=endrow.rateuom,
            previous2h_outside_stability_minutes=float((hi[bad]-lo[bad]).sum()/60),
            first_resume_source_minutes=float((later.iloc[0].starttime-t).total_seconds()/60) if len(later) else None,
            same_linkorderid_on_resume=bool(later.iloc[0].linkorderid==endrow.linkorderid) if len(later) else None,
            interpretation='RECORDED_INTERRUPTION_CONFIRMED_INTENT_UNRESOLVED')
        evidence.append(e)
    evidence=pd.DataFrame(evidence);evidence.to_csv(OUT/'source_record_evidence.csv',index=False)
    pd.DataFrame([dict(stay_id=sid,reason=';'.join(reasons),review_status='RECORD_TIMING_CHECKED_CLINICAL_INTENT_UNRESOLVED') for sid,reasons in selected.items()]).to_csv(OUT/'record_review_queue.csv',index=False)
    groups=[]
    for arm,g in details.groupby('A'):
        q=dict(arm=int(arm),n=len(g),n_past2h_stable=int(g.past2h_stable_at_action.sum()),
            n_low_dose_at_action=int(g.positive_low_dose_at_action.sum()),
            ne_median=float(g['ne'].median()),va_median=float(g.va.median()),delay_median_h=float(g.action_h.median()))
        for minutes in [5,15,30,120]:
            col=g[f'restart_within_{minutes}min']
            q[f'restart_{minutes}min_n']=int(col.eq(1).sum());q[f'unknown_{minutes}min_n']=int(col.isna().sum())
        groups.append(q)
    pd.DataFrame(groups).to_csv(OUT/'measurement_summary.csv',index=False)
    bb=balances[balances.weighting.eq('overlap_diagnostic')]
    summary=dict(status='DESIGN_DIAGNOSTICS_COMPUTED_NO_NEW_EFFECTS',groups=groups,
        overlap_max_absolute_smd=float(bb.smd.abs().max()),overlap_terms_above_point1=int(bb.smd.abs().gt(.1).sum()),
        overlap_ess=[r for r in ess if r['weighting']=='overlap_diagnostic'],new_effect_estimates_computed=False,
        record_review_n=len(evidence),clinical_intent_adjudicated=False,
        checks=dict(all_284_decisions=len(d)==284,all_470_pool=len(pool)==470,unique_stays=not details.stay_id.duplicated().any(),
            all_review_actions_have_matching_source_end=bool(evidence.source_end_matches_action.all()),
            timing_counts_reconcile=all(r['ne']+r['vaso']+r['simultaneous']+r['no_recorded_action_yet']==470 for r in timings),
            overlap_weights_finite_positive=bool(np.isfinite(ow).all() and (ow>0).all()),
            no_new_outcome_effect_columns_read=not bool({'event','time_h'}&set(d.columns))))
    assert all(summary['checks'].values())
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    g0,g1=groups
    timingtable=['| 距首次稳定 | NE先停 | 加压素先停 | 同时停 | 尚无记录动作 |','|---|---|---|---|---|']
    for row in timings:
        timingtable.append(f"| {row['hour']}小时 | {row['ne']} | {row['vaso']} | {row['simultaneous']} | {row['no_recorded_action_yet']} |")
    report=f'''# 下一阶段设计建议与可行性诊断

**建议：优先完成撤药记录复核，再开展明确标注为事后修订的重叠人群分析。CCW目前继续保持不可可靠解释，不通过更换截止时间寻找结果。**

本轮未计算新RR、RD或其他组间结局效应；保留原470次住院稳定池及284次实际撤药。诊断方案见[固定计划](../../docs/design_feasibility_plan_v1.md)。本轮不是独立验证或前瞻注册。

## 两组在停药前已经不同

| 项目 | NE先停（192次） | 加压素先停（92次） |
|---|---|---|
| 撤药前NE剂量中位数，mcg/kg/min | {g0['ne_median']:.3f} | {g1['ne_median']:.3f} |
| 撤药前加压素中位数，U/h | {g0['va_median']:.3f} | {g1['va_median']:.3f} |
| 距首次稳定的中位时间，h | {g0['delay_median_h']:.2f} | {g1['delay_median_h']:.2f} |
| 撤药前仍连续2小时符合低剂量稳定 | {g0['n_past2h_stable']}/192 | {g1['n_past2h_stable']}/92 |
| 15分钟内重启被撤药物 | {g0['restart_15min_n']}/192 | {g1['restart_15min_n']}/92 |
| 120分钟内重启被撤药物 | {g0['restart_120min_n']}/192 | {g1['restart_120min_n']}/92 |

这说明比较的不只是最终停哪种药，还包含此前不同的滴定状态；不能删除剂量变量来消除统计失衡。以上输注状态可观测性在所列重启窗口内完整，但短暂停顿仍可能是临床尝试或记录间隔，现有记录无法自动区分。用“停后必须持续两小时”排除这些人会重新引入未来信息选择。

## 重叠人群有进一步评估价值，但还没获得有效效应

使用原PS、未改模型，NE先停权重为PS，加压素先停为1-PS。48个协变量和缺失指标按同一固定未加权标准差分母计算，最大绝对SMD为{summary['overlap_max_absolute_smd']:.3f}，超过0.10的项目为{summary['overlap_terms_above_point1']}。两组ESS分别为{summary['overlap_ess'][0]['ess']:.1f}和{summary['overlap_ess'][1]['ess']:.1f}。

这是对已有模型平衡的诊断，不是“122个可互换患者”，ESS也不是实际样本数。0.099接近0.10，尚需检查分布形状、剂量轨迹和重抽样下的平衡稳定性。原PS使用惩罚回归，不能套用无惩罚Logistic最大似然的精确均值平衡性质。重叠权重强调可比较人群，会改变估计目标，不能回写原总体IPTW效应。

方法依据：[Li、Morgan、Zaslavsky，Balancing Covariates via Propensity Score Weighting](https://www.tandfonline.com/doi/full/10.1080/01621459.2016.1260466)。下一阶段应把目标写为实际撤药决策时的重叠人群校正关联，并保留未测混杂、滴定过程和测量误差限制；临床因果解释另需证明。

## 时间窗口分布

{chr(10).join(timingtable)}

分母均为原470次住院；“尚无记录动作”包括此前已经结束观察者，不是该时点仍在险人数。该表只描述既定12小时内动作，不用来选择最有利的宽限期；更早截止会改变策略及人群支持，不能宣称修复原CCW。

## 已准备的记录复核材料

将全部15分钟内重启、撤药前不再连续稳定的病例，以及每组10次固定随机比较病例列入复核队列，共{len(selected)}次。`review_source_infusion_rows.csv`保留相邻原始输注行，`review_timeline_segments.csv`提供动作前后两小时的重建分段。逐例记录时序核对已完成：全部撤药时刻均有对应药物正剂量源记录在同刻结束；逐例结束状态、续接时间、链接订单是否相同及稳定条件超限时长见`source_record_evidence.csv`。

`record_review_queue.csv`标记为`RECORD_TIMING_CHECKED_CLINICAL_INTENT_UNRESOLVED`。**记录时序已核对，临床停药意图仍不确定。** 相同链接订单的重启不能自动判为管路中断，不同链接订单也不能自动判为临床反跳。此复核未据此排除任何人。

建议按以下顺序推进：

1. 复核暂停/停止状态、续接/换速、剂量异常及是否重新加量后才停药，记录每个病例证据和不确定性。不能按哪个分组更有利来剔除病例。
2. 冻结新估计目标与主分析/敏感性：先保留284次实际撤药；若采用动作前两小时稳定限制，应明确是仅使用过去信息的不同人群敏感性，并重新检查支持度。复核完成后才重新拟合权重和效应，每次患者bootstrap重拟合全流程。
3. 比较反跳各组成、血压及NED时，保留观察终止、测量缺失与不同滴定状态的限制；不把机械性的剂量下降解释为临床获益。
4. 随后审计28天结局可观测性，覆盖再入ICU及死亡日期精度；不能沿用“出ICU以后全程无药”或“28减最后用药时间”的替代算法。外部数据库和全文更新放在定义稳定后。

计算状态：`DESIGN_DIAGNOSTICS_COMPUTED_NO_NEW_EFFECTS`。本轮检查确认人数、唯一标识、时间计数和权重有效性；不会将这些检查升级为因果假设验证。项目仍不可投稿。
'''
    (OUT/'next_stage_recommendation.md').write_text(report,encoding='utf-8')
    proc=subprocess.run([sys.executable,str(BASE/'scripts/test_design_feasibility.py')],capture_output=True,text=True)
    (OUT/'unit_tests.txt').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    checks=summary['checks'].copy();checks['unit_tests_pass']=proc.returncode==0
    checks['input_manifest_matches']=all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in manifests)
    prior=[]
    for file in [SRC/'core_manifest.json',SRC/'weight_time_audit/manifest.json']:
        for r in json.loads(file.read_text(encoding='utf-8')):
            path=Path(r['path']);path=path if path.is_absolute() else BASE/path
            prior.append(hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'])
    checks['previous_analysis_manifests_unchanged']=all(prior)
    validation=dict(status='DESIGN_FEASIBILITY_CHECKS_PASS' if all(checks.values()) else 'DESIGN_FEASIBILITY_CHECKS_FAILED',
        checks=checks,clinical_intent_adjudicated=False,new_effect_estimates_computed=False,submission_ready=False)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    produced=list(OUT.glob('*'))+[Path(__file__),BASE/'scripts/test_design_feasibility.py',BASE/'docs/design_feasibility_plan_v1.md']
    output_manifest=[dict(path=str(p.relative_to(BASE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in produced if p.is_file() and p.name!='manifest.json']
    (OUT/'manifest.json').write_text(json.dumps(output_manifest,indent=2),encoding='utf-8')
    assert all(checks.values()),checks
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
