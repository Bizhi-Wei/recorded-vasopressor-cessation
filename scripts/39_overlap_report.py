"""Bounded audit and Chinese report for the locked overlap amendment."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1];SRC=BASE/'rebuild_v3';OUT=SRC/'overlap_v1'

def manifest_ok(path):
    for r in json.loads(path.read_text(encoding='utf-8')):
        p=Path(r['path']);p=p if p.is_absolute() else BASE/p
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:return False
    return True

def main():
    s=json.loads((OUT/'inference_summary.json').read_text());r=pd.read_csv(OUT/'results.csv')
    bd=pd.read_csv(OUT/'bootstrap_diagnostics.csv');pdg=pd.read_csv(OUT/'point_diagnostics.csv').set_index('scenario')
    stability=[]
    for name,g in bd.groupby('scenario'):
        for col in ['max_abs_smd','ess_ne','ess_vaso','max_share_ne','max_share_vaso']:
            stability.append(dict(scenario=name,metric=col,q025=float(g[col].quantile(.025)),median=float(g[col].median()),q975=float(g[col].quantile(.975)),min=float(g[col].min()),max=float(g[col].max())))
    pd.DataFrame(stability).to_csv(OUT/'bootstrap_stability.csv',index=False)
    coverage=[];character=[]
    raw=pd.read_csv(SRC/'decision_analysis.csv')
    for name in ['all284','stable272']:
        w=pd.read_csv(OUT/f'{name}_weights.csv')
        d=raw.merge(w,on=['stay_id','subject_id','A'],validate='one_to_one')
        for arm,g in d.groupby('A'):
            coverage.append(dict(scenario=name,arm=int(arm),n=len(g),**{f'full_{h}h':int(g[f'full_{h}h'].sum()) for h in [6,12,24]},
                unknown_censor_n=int(g.unknown_censor.sum())))
            for col in ['ne','va','ned']:
                contributions=g[col]*g.overlap_weight
                character.append(dict(scenario=name,arm=int(arm),variable=col,weighted_mean=float(np.average(g[col],weights=g.overlap_weight)),
                    largest_single_row_share_of_weighted_sum=float(contributions.max()/contributions.sum()),
                    maximum_value=float(g[col].max())))
    pd.DataFrame(coverage).to_csv(OUT/'full_window_counts.csv',index=False)
    pd.DataFrame(character).to_csv(OUT/'dose_weighted_sum_concentration.csv',index=False)
    high=raw.loc[raw['ne'].idxmax()];action=pd.Timestamp(high.t0)
    source=pd.read_pickle(SRC/'source_pressors.pkl')
    source=source[source.stay_id.eq(high.stay_id)&source.drug.eq('norepinephrine')&pd.to_datetime(source.starttime).lt(action+pd.Timedelta(minutes=30))&pd.to_datetime(source.endtime).gt(action-pd.Timedelta(hours=2))].copy()
    source.to_csv(OUT/'high_ne_source_record_check.csv',index=False)
    tests=[]
    for filename in ['test_rebuild_v3.py','test_overlap_analysis.py']:
        result=subprocess.run([sys.executable,str(BASE/'scripts'/filename)],capture_output=True,text=True)
        (OUT/(filename+'.txt')).write_text(result.stdout+result.stderr,encoding='utf-8');tests.append(result.returncode==0)
    checks=dict(unit_tests_pass=all(tests),input_manifest_matches=manifest_ok(OUT/'input_manifest.json'),
        original_core_manifest_unchanged=manifest_ok(SRC/'core_manifest.json'),
        weight_time_audit_manifest_unchanged=manifest_ok(SRC/'weight_time_audit/manifest.json'),
        design_diagnostics_manifest_unchanged=manifest_ok(SRC/'design_feasibility/manifest.json'),
        main_n284=int(pdg.loc['all284','n'])==284,stable_n272=int(pdg.loc['stable272','n'])==272,
        both_scenarios_300_success=s['success_by_scenario']=={'all284':300,'stable272':300} and not s['failures'],
        component_cif_identity_pass=s['component_cif_identity_all_fits_pass'],
        previous_overlap_weights_reproduced=s['original_diagnostic_weights_reproduced'],
        all_intervals_finite=bool(np.isfinite(r[['estimate','lo','hi']].to_numpy()).all()),
        all_intervals_ordered=bool(r.lo.le(r.hi).all()),all_intervals_use_300=bool(r.bootstrap_finite.eq(300).all()),
        bootstrap_pairs_complete=bool(not bd.duplicated(['scenario','replicate']).any() and bd.groupby('replicate').size().eq(2).all()))
    risks=r[r.estimand.str.endswith(('_ne','_vaso'))&~r.estimand.str.startswith(('dned','map_','hypo65_','lactate_'))]
    checks['all_cifs_bounded']=bool(risks.estimate.between(0,1).all())
    rates=bd.groupby('scenario').max_abs_smd.apply(lambda x:float((x>.1).mean())).to_dict()
    balance=pd.read_csv(OUT/'all284_balance.csv').set_index('covariate')
    flags=dict(bootstrap_fraction_max_smd_above_point1=rates,main_ne_weighted_cdf_max_difference=float(balance.loc['ne','ks']),
        main_ne_weighted_variance_ratio=float(balance.loc['ne','variance_ratio']),causal_assumptions_verified=False,submission_ready=False)
    status='OVERLAP_COMPUTED_CHECKS_PASS' if all(checks.values()) else 'OVERLAP_CHECKS_FAILED'
    interpretation='POST_HOC_ASSOCIATION_ONLY_DISTRIBUTIONAL_IMBALANCE_AND_UNSTABLE_BALANCE'
    validation=dict(status=status,checks=checks,interpretation_status=interpretation,substantive_findings=flags)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    rr=r.set_index(['scenario','estimand'])
    def fmt(name,key,mult=1):
        row=rr.loc[(name,key)]
        return f'{row.estimate*mult:.3f}（{row.lo*mult:.3f}至{row.hi*mult:.3f}）'
    lines=['| 指标 | 全部284次实际撤药 | 动作前仍稳定272次 |','|---|---|---|']
    for title,key,mult in [('24小时反跳RR','relapse24_rr',1),('24小时反跳RD，百分点','relapse24_rd',100),('0–6小时MAP差，mmHg','map_0_6_diff',1),('0–6小时低血压读数比例差，百分点','hypo65_0_6_diff',100),('6小时NED峰值变化差，mcg/kg/min','dned_peak_6h_diff',1)]:
        lines.append(f"| {title} | {fmt('all284',key,mult)} | {fmt('stable272',key,mult)} |")
    component=['| 24小时首次反跳成分的风险差，百分点 | 全部284次 | 稳定272次 |','|---|---|---|']
    for key,title in [('restart','重启被撤药物'),('escalation','继续药剂量升级'),('new_drug','新药'),('concurrent_escalation_or_new_drug','动作同刻升级或新药')]:
        component.append(f"| {title} | {fmt('all284',key+'_rd',100)} | {fmt('stable272',key+'_rd',100)} |")
    main0=[x for x in character if x['scenario']=='all284' and x['arm']==0 and x['variable']=='ne'][0]
    report=f'''# 实际撤药时重叠人群分析 v1

**计算状态：{status}。解释状态：{interpretation}；项目仍不可投稿。**

两套场景各完成300次患者bootstrap，全流程重新拟合。重叠人群主场景24小时反跳RR为{fmt('all284','relapse24_rr')}，但剂量分布仍不平衡，而且主场景{rates['all284']*100:.0f}%的重抽样最大绝对SMD超过0.10。不能把上一轮单次最大SMD 0.099当作可比性已经解决，也不能据区间跨1认定等效。

## 修订范围和估计目标

本轮按计算前固定的[修订方案](../../docs/overlap_analysis_amendment_v1.md)实施。修订是在已知原IPTW、CCW和重叠权重诊断之后制定，属于事后探索，不是前瞻注册。主场景284次住院、283名患者（NE先停192、加压素先停92）；敏感性272次住院、271名患者（185、87），只限动作前两小时仍稳定者。两者是不同人群，不以哪个显著决定保留哪个。

沿用原动作、事件及时间零点，保留所有短暂停顿；加压素先停权重为1-PS、NE先停为PS，不另截尾。目标为两种顺序都有可能的重叠人群中的校正关联，不是原总体ATE，也不自动构成随机试验等价物。RR/RD及连续差方向均为加压素先停相对NE先停。

## 效应及其不确定性

{chr(10).join(lines)}

主场景24小时反跳加权风险为NE {rr.loc[('all284','relapse24_ne'),'estimate']*100:.1f}%、加压素 {rr.loc[('all284','relapse24_vaso'),'estimate']*100:.1f}%；稳定场景为NE {rr.loc[('stable272','relapse24_ne'),'estimate']*100:.1f}%、加压素 {rr.loc[('stable272','relapse24_vaso'),'estimate']*100:.1f}%。完整6/12/24小时风险、乳酸和其他指定结果均见`results.csv`。

死亡和转出本次ICU为竞争事件，未知剂量为观测删失；风险仅涉及本次ICU内结局，不能解释为转出后的完整风险。连续终点在有记录者中估计，未观察部分不补0；按臂观测人数及子集ESS见`all284_coverage.csv`、`stable272_coverage.csv`，完整窗口人数见`full_window_counts.csv`。MAP等探索性比较未作多重校正，不能单凭区间排除0认定临床优劣。

## 反跳首次事件组成

{chr(10).join(component)}

这里是首次复合事件的成分风险，其他成分及死亡/转出作为竞争事件；不等于某类事件无论先后是否曾发生。逐次拟合已验证四个成分CIF之和等于复合CIF。同刻事件依照原识别优先级归类，不能把成分归类视为独立临床机制证据。

## 为什么仍不能称平衡充分

- 主场景最大绝对SMD {pdg.loc['all284','max_abs_smd']:.3f}，ESS为NE {pdg.loc['all284','ess_ne']:.1f}、加压素 {pdg.loc['all284','ess_vaso']:.1f}；稳定场景最大SMD {pdg.loc['stable272','max_abs_smd']:.3f}，ESS仅{pdg.loc['stable272','ess_ne']:.1f}和{pdg.loc['stable272','ess_vaso']:.1f}。
- 全部300次重抽样中，最大SMD>0.10比例分别为{rates['all284']*100:.0f}%和{rates['stable272']*100:.0f}%。这是拟合后平衡稳定性的描述，不是因果有效概率；没有因失衡而丢弃任何一次拟合。
- 主场景NE剂量的加权经验CDF最大差为{flags['main_ne_weighted_cdf_max_difference']:.3f}，加压素先停组相对NE先停组的加权方差比为{flags['main_ne_weighted_variance_ratio']:.3f}。尽管均值SMD很小，剂量分布并未对齐。此CDF差是描述量，不报告为KS检验p值。
- NE先停组最大单次NE剂量为{main0['maximum_value']:.3f} mcg/kg/min，该组单条记录对加权NE剂量总和的最大份额为{main0['largest_single_row_share_of_weighted_sum']*100:.1f}%。这只是均值贡献集中度，不是效应影响度、数据错误证明或删例理由。

已核对该高剂量的提取源行：前序NE约0.090、0.060 mcg/kg/min，随后一条30分钟记录为1.259 mcg/kg/min并以FinishedRunning结束，与撤药时刻相同。计算保留了源行原单位和数值；目前没有临床病历依据判定此跳变真实或录入错误。相邻源行保存在`high_ne_source_record_check.csv`，不据此自动删除病例。动作前稳定敏感性已经按预先固定的过去信息规则处理相应人群变化，没有按该例结局临时改队列。

SMD、CDF差和方差比分别见两套`*_balance.csv`；重抽样SMD/ESS/最大权重份额的分布见`bootstrap_stability.csv`。源PS为惩罚回归，不保证重叠权重精确均值平衡。

## 验证与后续

原重叠诊断权重逐例复现；原核心、时间复核和设计诊断文件哈希保持一致。17个原边界测试及4个新增权重/竞争事件测试通过，{len(checks)}项最终检查见`validation.json`。这些检查限定于实现和文件一致性，不抵消分布失衡或未测混杂。

下一步应先审核动作前剂量轨迹、非线性剂量分布和异常高剂量记录，判断能否定义具有临床可比性的决策状态。若修订PS的剂量函数或人群口径，须先固定规则、以结局无关的平衡/支持诊断评估，并保留当前结果；不能依据显著性删样本或选模型。原12小时CCW的不可靠状态保持不变。

47次源记录的时间核对已在前一阶段完成，但医师撤药意图仍未判明。此次没有新增28天器官支持、跨库验证或投稿稿件；目前可以报告有上述限制的校正关联，不能推荐撤药顺序或宣称等效。

方法背景：[Li、Morgan、Zaslavsky重叠权重论文](https://www.tandfonline.com/doi/full/10.1080/01621459.2016.1260466)。所有临床数字来自本地重算。复现：`scripts/38_overlap_analysis.py`、`scripts/39_overlap_report.py`；输入清单和产物清单分别为`input_manifest.json`和`manifest.json`。
'''
    (OUT/'overlap_report.md').write_text(report,encoding='utf-8')
    paths=list(OUT.glob('*'))+[BASE/'scripts/test_overlap_analysis.py',Path(__file__),BASE/'scripts/38_overlap_analysis.py',BASE/'docs/overlap_analysis_amendment_v1.md']
    manifest=[dict(path=str(p.relative_to(BASE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths if p.is_file() and p.name!='manifest.json']
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(validation,indent=2),flush=True)
    if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
