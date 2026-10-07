"""Audit completed core rebuild and produce a bounded, reproducible local report."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'rebuild_v3'


def main():
    summary=json.loads((OUT/'inference_summary.json').read_text())
    cohort=json.loads((OUT/'cohort_summary.json').read_text())
    datasets=json.loads((OUT/'dataset_summary.json').read_text())
    d=pd.read_csv(OUT/'decision_analysis.csv')
    l=pd.read_csv(OUT/'ccw_long.csv');c=pd.read_csv(OUT/'ccw_clones.csv')
    w=pd.read_csv(OUT/'ccw_weights.csv')
    dr=pd.read_csv(OUT/'decision_results.csv').set_index('estimand')
    cr=pd.read_csv(OUT/'ccw_results.csv').set_index('estimand')
    pool=pd.read_csv(OUT/'stable_pool.csv')
    tests=subprocess.run([sys.executable,str(BASE/'scripts/test_rebuild_v3.py')],capture_output=True,text=True)
    (OUT/'unit_test_output.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8')
    checks=dict(unit_tests_pass=tests.returncode==0,
        decision_from_same_stable_pool=set(d.stay_id).issubset(pool.stay_id),
        decision_time_not_before_stable=bool((pd.to_datetime(d.t0)>=pd.to_datetime(d.stable_t0)).all()),
        both_strategy_codes_present=set(w.arm)=={0,1},
        exactly_two_clones_per_stay=bool(c.groupby('stay_id').size().eq(2).all()),
        no_duplicate_person_hour=not l.duplicated(['stay_id','arm','block']).any(),
        no_double_terminal_events=bool((l.groupby(['stay_id','arm'])[['target','compete','censor','obs_censor']].sum().sum(axis=1)<=1).all()),
        followup_bounded=bool(d.time_h.between(0,24).all()),
        outcome_cifs_are_probabilities=bool(dr.loc[[x for x in dr.index if x.startswith(('iptw_relapse','unweighted_relapse','untrimmed_relapse')) and x.endswith(('_ne','_vaso'))],'estimate'].between(0,1).all()),
        ccw_probabilities_bounded=bool(cr.loc[[x for x in cr.index if x.endswith(('_ne','_vaso'))],'estimate'].between(0,1).all()),
        positive_finite_weights=bool(np.isfinite(w.weight).all()&(w.weight>0).all()),
        no_failed_decision_bootstraps=summary['decision_bootstrap_success']==summary['bootstrap_requested'],
        no_failed_ccw_bootstraps=summary['ccw_bootstrap_success']==summary['bootstrap_requested'])
    # Risk rows must stop immediately after the first terminating interval.
    for _,g in l.groupby(['stay_id','arm']):
        terminal=g[['target','compete','censor','obs_censor']].sum(axis=1)>0
        if terminal.any():
            checks['no_rows_after_terminal']=checks.get('no_rows_after_terminal',True) and int(g.loc[terminal,'block'].iloc[0])==int(g.block.max())
    deadline=c[(c.terminal==3)&c.stop_h.eq(12)]
    checks['deadline_at12_not_lost']=len(deadline)==len(l[(l.censor==1)&l.block.eq(11)&l.stop_h.eq(12)])
    # Once adherent, cumulative IPCW must remain fixed (before arm truncation).
    for _,g in w.groupby('clone_id'):
        x=g[g.pending==0].weight_raw
        if len(x)>1:checks['weights_fixed_after_adherence']=checks.get('weights_fixed_after_adherence',True) and bool(np.allclose(x,x.iloc[0]))
    cohort_hash=hashlib.sha256((OUT/'stable_pool.csv').read_bytes()).hexdigest()
    metas=[json.loads((OUT/f'raw_{name}_meta.json').read_text()) for name in ['chart','lab','urine','procedure']]
    checks['raw_extractions_match_final_cohort']=all(x['signature']['cohort_sha256']==cohort_hash for x in metas)
    checks['dataset_build_checks_pass']=all(datasets['checks'].values())
    for label,table in [('decision',dr),('ccw',cr)]:
        checks[label+'_finite_estimates_and_intervals']=bool(np.isfinite(table[['estimate','lo','hi']].to_numpy()).all())
        checks[label+'_ordered_intervals']=bool(table.lo.le(table.hi).all())
        checks[label+'_all_intervals_use_requested_replicates']=bool(table.bootstrap_finite.eq(summary['bootstrap_requested']).all())
    substantive_flags=dict(
        decision_residual_imbalance=summary['decision']['max_smd_after']>0.1,
        decision_extreme_propensity=summary['decision']['ps_min']<0.01 or summary['decision']['ps_max']>0.99,
        decision_truncation_changes_rd_direction=bool(dr.loc['iptw_relapse24_rd','estimate']*dr.loc['untrimmed_relapse24_rd','estimate']<0),
        ccw_hourly_approximation=True,
        causal_assumptions_verified=False)
    versions=dict(python=sys.version,executable=sys.executable,packages={name:importlib.metadata.version(name) for name in ['pandas','numpy','scipy','scikit-learn','duckdb','threadpoolctl']})
    (OUT/'runtime_versions.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    status='CORE_REBUILD_COMPUTED_CHECKS_PASS' if all(checks.values()) else 'CORE_REBUILD_CHECKS_FAILED'
    missing=d[datasets['covariates']].isna().mean().rename('missing_fraction').rename_axis('covariate').reset_index()
    missing.to_csv(OUT/'baseline_missingness.csv',index=False)
    outcome_columns=[x for x in d.columns if x.startswith(('dned_','ned_mean_','map_','hypo65_','lactate_')) and x not in datasets['covariates']]
    coverage=[]
    for arm,g in d.groupby('A'):
        for col in outcome_columns:
            coverage.append(dict(arm=int(arm),outcome=col,n=len(g),n_observed=int(g[col].notna().sum()),missing_fraction=float(g[col].isna().mean())))
    pd.DataFrame(coverage).to_csv(OUT/'outcome_coverage.csv',index=False)
    components=d[d.event.eq(1)].groupby(['A','event_reason']).size().rename('n_first_events').reset_index()
    components.to_csv(OUT/'decision_event_components.csv',index=False)
    manifest=[]
    paths=list((BASE/'scripts').glob('*rebuild*.py'))+[BASE/'docs/rebuild_protocol_v3_20260914.md']
    paths+=list(OUT.glob('*.csv'))+list(OUT.glob('*summary.json'))+list(OUT.glob('*meta.json'))
    paths+=list(OUT.glob('*.pkl'))+[OUT/'cohort_input_manifest.json',OUT/'extraction_manifest.json',OUT/'runtime_versions.json']
    for p in sorted(set(paths)):
        with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        manifest.append(dict(path=str(p.relative_to(BASE)),sha256=sha,bytes=p.stat().st_size))
    (OUT/'core_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT/'validation.json').write_text(json.dumps(dict(status=status,checks=checks,substantive_flags=substantive_flags,causal_assumptions_verified=False,submission_ready=False),indent=2),encoding='utf-8')
    def fmt(table,key,mult=1):
        r=table.loc[key]
        return f"{r.estimate*mult:.3f}（{r.lo*mult:.3f}至{r.hi*mult:.3f}）"
    points=[]
    for label,table,key in [('实际撤药后24h ICU内反跳RR',dr,'iptw_relapse24_rr'),('实际撤药后24h ICU内反跳RD 百分点',dr,'iptw_relapse24_rd'),('CCW 24h支持升级或死亡RR',cr,'ccw_rr'),('CCW 24h支持升级或死亡RD 百分点',cr,'ccw_rd'),('6h累计窗口NED峰值变化差',dr,'dned_peak_6h_diff'),('0–6h MAP差 mmHg',dr,'map_0_6_diff'),('0–6h低血压读数比例差 百分点',dr,'hypo65_0_6_diff')]:
        points.append(f"| {label} | {fmt(table,key,100 if '百分点' in label else 1)} |")
    sd,sc=summary['decision'],summary['ccw']
    report=f'''# MIMIC心脏术后撤药顺序核心重建结果

日期：2026-09-14。**状态：{status}；项目仍不可投稿。** 这表示本轮核心代码及列出的数据检查完成，不代表因果假设已经验证，也不替代独立方法学复核。

**当前主要问题是实际撤药组间仍不平衡，而且权重截尾会改变RD点估计方向。** 因此下列效应属于审计后重算结果，不能据此确定优先撤哪种药，也不能以区间跨越无效线认定策略等效。

## 本轮做了什么

按[重建方案](../docs/rebuild_protocol_v3_20260914.md)重建明确手术日期之后的稳定状态池、原始精度输注段、实际撤药结局、每小时CCW风险集、仅过去信息的生理协变量和权重。修正Paused区间语义；不再用未来停药持续时间选择人群，不再用整段住院联用时间或出院诊断作为基线。原始数据和旧结果未覆盖。

## 队列

- 严格手术编码成年ICU stays：{cohort['strict_surgery_adult_stays']}。
- 首次连续2小时低剂量联用且已过手术日期的稳定池：{cohort['stable_pool_n']}次住院，{cohort['stable_subjects']}名患者。
- 12小时内首次单药撤除候选：{cohort['decision_pool_n']}；NE先停{cohort['decision_groups'].get('norepinephrine',0)}，加压素先停{cohort['decision_groups'].get('vasopressin',0)}。同时撤药{cohort['simultaneous_n']}；无动作{cohort['no_action_within12h_n']}。
- 经可观测性检查，实际撤药模型：{len(d)}例（NE先停{sd['n_ne']}，加压素先停{sd['n_vaso']}）。CCW：{datasets['ccw_pool_n']}次住院、{datasets['ccw_clones']}个克隆。
- 实际撤药队列6/12/24小时完整ICU观测数：{datasets['decision_full_windows']}；剂量未知导致的观测删失：{datasets['unknown_censor_n']}例。
- 新旧队列不同，旧393例的效应不能与新效应当作相同人群的直接前后对比；中间版本427/257例因Paused修正已被替代。

## 两种分析必须分开解释

实际撤药分析从首次单药中断时刻起算，报告本次ICU内反跳的累积发生风险，死亡/转出为竞争事件。CCW从首次稳定状态起算，比较12小时内不同初始撤药顺序策略下的支持升级或死亡风险；停药前的剂量升级属于该结局，不叫撤药后反跳。CCW定义发生了审计后修订，不作为旧分析的同结局敏感性验证。

所有RR/RD方向均为**加压素先停相对NE先停**。连续终点使用实际可观测ICU窗口，未观察部分不补零；其缺失/转出选择限制仍需保留。患者聚类bootstrap每场景{summary['bootstrap_requested']}次，每次重新拟合插补、标准化及相应PS或删失模型。

| 指标 | 点估计及95% bootstrap区间 |
|---|---|
{chr(10).join(points)}

24h实际撤药IPTW风险：NE {dr.loc['iptw_relapse24_ne','estimate']*100:.1f}%，加压素 {dr.loc['iptw_relapse24_vaso','estimate']*100:.1f}%。CCW相应风险：NE {cr.loc['ccw_ne','estimate']*100:.1f}%，加压素 {cr.loc['ccw_vaso','estimate']*100:.1f}%。完整结果见`decision_results.csv`和`ccw_results.csv`，其中包含未加权、未截尾权重及不同随访窗口结果。

各连续终点按策略的实际观测人数及缺失比例见`outcome_coverage.csv`；基线缺失比例见`baseline_missingness.csv`。连续终点差异限于有记录的窗口，不能解释为全体患者的完整24小时生理效应。

复合反跳结局的首次事件组成见`decision_event_components.csv`，应核对重启、继续药升级和新药三类是否呈现不同关联；这些原始计数仅作描述，不是校正后的成分效应。

## 权重和验证

- 实际撤药IPTW ESS {sd['ess']:.1f}/{sd['n']}，分臂NE {sd['ess_ne']:.1f}、加压素 {sd['ess_vaso']:.1f}；最大绝对SMD由{sd['max_smd_before']:.3f}到{sd['max_smd_after']:.3f}，仍有{sd['covariates_above_0_1']}项>0.10。这些残留失衡必须随结果报告。
- PS范围{sd['ps_min']:.6f}–{sd['ps_max']:.6f}，有接近0/1的预测。未截尾24h反跳RD为{fmt(dr,'untrimmed_relapse24_rd',100)}个百分点，截尾后为{fmt(dr,'iptw_relapse24_rd',100)}个百分点；点估计方向变化提示对权重处理敏感。这里只按既定方案展示，没有按结果改模型。
- CCW原始权重最大{sc['weight_raw_max']:.2f}，99%截尾后最大{sc['weight_max']:.2f}；最低分臂小时风险集ESS {sc['min_block_ess']:.1f}。按小时建模是近似，依赖删失模型与支持度。
- 实际撤药bootstrap成功{summary['decision_bootstrap_success']}/{summary['bootstrap_requested']}；CCW成功{summary['ccw_bootstrap_success']}/{summary['bootstrap_requested']}。失败原因保存在`inference_summary.json`，不静默补齐。
- 所有源提取均与最终稳定池SHA256匹配；代码边界测试、队列包含关系、12小时边界删失、事件后无风险行、完成策略后权重不再更新等检查见`validation.json`。

## 仍然不能声称的结论

不能把检查通过视作无未测混杂、策略等效、没有代谢后果或临床优劣获证。严格手术日期口径舍弃了手术当日及仅服务代理病例，外推对象变窄。输注缺失与真实停药仍不能完全区分；剂量unknown区间作为观测删失，依赖观测机制假设；按记录重建不能等同实时已知信息。乳酸和其他生理指标的选择性测量仍在。残余平衡、权重敏感性、1小时CCW离散近似应进一步审阅。

本轮未重建28天器官支持结局、全部旧敏感性、eICU/SICdb外部分析，也未重生成投稿Word/PDF。旧论文及三库结论继续处于待修订状态。

## 复现

依次运行`scripts/29_rebuild_cohort.py`、`30_rebuild_extract.py`、`31_rebuild_datasets.py`、`32_rebuild_inference.py`、`33_rebuild_verify_report.py`。大表缓存使用队列SHA256、源文件大小/时间与所需列标识；分析产物及代码哈希见`core_manifest.json`。依赖pandas、numpy、scipy、scikit-learn、duckdb和threadpoolctl。

输入事件语义参考[MIMIC官方inputevents说明](https://mimic.mit.edu/docs/iv/modules/icu/inputevents.html)，时间零点与NEE来源见重建方案。临床结果来自本地重分析，不来自文献推算。
'''
    (OUT/'core_rebuild_report.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(status=status,checks=checks),indent=2),flush=True)
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
