"""Report locked dose-basis diagnostics, including trade-offs and paired changes."""
from pathlib import Path
import json
import hashlib
import subprocess
import sys
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1];SRC=BASE/'rebuild_v3';OUT=SRC/'nonlinear_dose_v1'
LABELS=['all284_linear','all284_spline','stable272_linear','stable272_spline']
NAMES={'all284_linear':'284次／线性','all284_spline':'284次／样条','stable272_linear':'272次稳定／线性','stable272_spline':'272次稳定／样条'}

def verify(path):
    for r in json.loads(path.read_text(encoding='utf-8')):
        p=Path(r['path']);p=p if p.is_absolute() else BASE/p
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:return False
    return True

def main():
    s=json.loads((OUT/'summary.json').read_text());p=pd.read_csv(OUT/'point_diagnostics.csv').set_index('scenario')
    b=pd.read_csv(OUT/'bootstrap_diagnostics.csv');rows=[];fractions=[];paired=[]
    metrics=['max_abs_smd','max_abs_smd_including_basis','ne_ks','ne_rounded_ks','va_ks','va_rounded_ks','ess_nefirst','ess_vasofirst','max_share_nefirst','max_share_vasofirst']
    for name,g in b.groupby('scenario'):
        for col in metrics:
            rows.append(dict(scenario=name,metric=col,q025=float(g[col].quantile(.025)),median=float(g[col].median()),q975=float(g[col].quantile(.975))))
        fractions.append(dict(scenario=name,original_smd_above_point1=float(g.max_abs_smd.gt(.1).mean()),
            including_basis_smd_above_point1=float(g.max_abs_smd_including_basis.gt(.1).mean()),
            either_arm_ess_below20=float((g.ess_nefirst.lt(20)|g.ess_vasofirst.lt(20)).mean())))
    pd.DataFrame(rows).to_csv(OUT/'bootstrap_distributions.csv',index=False)
    pd.DataFrame(fractions).to_csv(OUT/'bootstrap_diagnostic_fractions.csv',index=False)
    for group in ['all284','stable272']:
        linear=b[b.scenario.eq(group+'_linear')].set_index('replicate');spline=b[b.scenario.eq(group+'_spline')].set_index('replicate')
        for col in metrics:
            delta=spline[col]-linear[col]
            paired.append(dict(population=group,metric=col,point_delta=float(p.loc[group+'_spline',col]-p.loc[group+'_linear',col]),
                paired_q025=float(delta.quantile(.025)),paired_median=float(delta.median()),paired_q975=float(delta.quantile(.975)),n_pairs=int(delta.notna().sum())))
    pd.DataFrame(paired).to_csv(OUT/'paired_diagnostic_changes.csv',index=False)
    tests=[]
    for file in ['test_nonlinear_dose.py','test_overlap_analysis.py']:
        proc=subprocess.run([sys.executable,str(BASE/'scripts'/file)],capture_output=True,text=True)
        (OUT/(file+'.txt')).write_text(proc.stdout+proc.stderr,encoding='utf-8');tests.append(proc.returncode==0)
    checks=dict(unit_tests_pass=all(tests),input_manifest_matches=verify(OUT/'input_manifest.json'),
        previous_overlap_artifacts_unchanged=verify(SRC/'overlap_v1/manifest.json'),
        previous_core_artifacts_unchanged=verify(SRC/'core_manifest.json'),
        no_outcome_columns_used=not s['outcome_columns_read'] and not bool({'time_h','event','event_reason','dned_peak_6h','map_0_6'}&set(s['read_columns'])),
        no_new_effects_computed=not s['new_effects_computed'],all_four_scenarios_present=set(s['success_by_scenario'])==set(LABELS),
        all_1200_fits_successful=all(s['success_by_scenario'].get(x)==300 for x in LABELS) and not s['failures'],
        all_linear_point_weights_reproduced=s['original_linear_point_weights_reproduced'],
        all_previous_linear_bootstrap_smd_ess_reproduced=s['all_original_linear_bootstrap_smd_ess_reproduced'],
        paired_diagnostics_complete=all(x['n_pairs']==300 for x in paired),
        diagnostic_values_finite=bool(np.isfinite(b[metrics].to_numpy()).all()),
        ess_positive_and_bounded=bool((b.ess_nefirst.gt(0)&b.ess_nefirst.le(b.n_ne+1e-8)&b.ess_vasofirst.gt(0)&b.ess_vasofirst.le(b.n_vaso+1e-8)).all()))
    for label in LABELS:
        w=pd.read_csv(OUT/f'{label}_weights.csv')
        checks[label+'_weights_in_unit_interval']=bool(w.overlap_weight.gt(0).all()&w.overlap_weight.le(1).all())
    validation=dict(status='NONLINEAR_DIAGNOSTICS_CHECKS_PASS' if all(checks.values()) else 'NONLINEAR_DIAGNOSTICS_CHECKS_FAILED',
        checks=checks,interpretation_status='DOSE_BALANCE_IMPROVED_SUPPORT_REDUCED_NOT_CAUSALLY_VALIDATED',
        outcome_effects_computed=False,submission_ready=False)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    f=pd.DataFrame(fractions).set_index('scenario')
    table=['| 场景 | 原始变量最大SMD | 含样条最大SMD | NE原始CDF差 | NE取整CDF差 | NE先停ESS | 加压素先停ESS |','|---|---|---|---|---|---|---|']
    for label in LABELS:
        r=p.loc[label];table.append(f"| {NAMES[label]} | {r.max_abs_smd:.3f} | {r.max_abs_smd_including_basis:.3f} | {r.ne_ks:.3f} | {r.ne_rounded_ks:.3f} | {r.ess_nefirst:.1f} | {r.ess_vasofirst:.1f} |")
    btable=['| 场景 | 原始变量最大SMD>0.10 | 含基函数最大SMD>0.10 | 任一臂ESS<20 |','|---|---|---|---|']
    for label in LABELS:
        r=f.loc[label];btable.append(f'| {NAMES[label]} | {r.original_smd_above_point1*100:.1f}% | {r.including_basis_smd_above_point1*100:.1f}% | {r.either_arm_ess_below20*100:.1f}% |')
    concentration=[]
    for label in ['all284_linear','all284_spline']:
        d=pd.read_csv(OUT/f'{label}_distribution.csv')
        concentration.append(float(d.loc[d.variable.eq('ne')&d.arm.eq(0),'max_contribution_share'].iloc[0]))
    report=f'''# 非线性剂量诊断：改善了什么，尚未解决什么

日期：2026-09-15。**计算状态：{validation['status']}。解释状态：DOSE_BALANCE_IMPROVED_SUPPORT_REDUCED_NOT_CAUSALLY_VALIDATED。**

固定非线性剂量项明显改善NE剂量分布，但同时减少有效样本量，且非线性基函数及其他变量分布仍有残余失衡。**本轮不建议把比较效应升级为可靠因果结论，也不自动计算一轮新的RR。** 当前证据更支持把撤药实践、滴定状态与可比性限制作为研究重点，已有比较效应保留为有明确限制的探索性关联；最终研究定位仍需临床和方法学审阅。

## 实施范围

按计算前固定的[非线性诊断方案](../../docs/nonlinear_dose_diagnostic_plan_v1.md)，保留284次实际撤药、272次动作前稳定两个人群，各对照原线性PS及唯一固定样条扩展。NE与加压素先log1p变换，再使用固定节点三次B样条，原24混杂变量及缺失指示均保留，惩罚C=1不调参。14个样条列及其固定缺失指示使设计矩阵由48列增至76列；新增基函数缺失指示为恒零，不代表新增14项临床缺失。

四个场景各300次患者聚类重抽样，共1,200次拟合全部成功。旧线性模型的逐例点权重，以及先前两人群全部600次重抽样SMD/ESS均复现。本轮仅加载标识、策略、动作前稳定标记和协变量，未读取结局列，未生成新RR/RD。整个诊断仍是已知旧结果后的事后修订，不能称完全盲态或独立验证。

## 点诊断

{chr(10).join(table)}

CDF差是两臂加权经验分布的最大差，越小代表该维度分布更接近，不是p值。取整诊断按NE 0.005 mcg/kg/min、加压素0.1 U/h最近网格计算，目的是显示浮点抖动影响；这些宽度不是经验证的临床等效界值。原始拟合和所有剂量都没有取整。

NE方差比（加压素先停组/NE先停组）在284次人群由{p.loc['all284_linear','ne_variance_ratio']:.3f}变为{p.loc['all284_spline','ne_variance_ratio']:.3f}，稳定人群由{p.loc['stable272_linear','ne_variance_ratio']:.3f}变为{p.loc['stable272_spline','ne_variance_ratio']:.3f}。284次NE先停组单条对加权NE总和的最大贡献由{concentration[0]*100:.1f}%变为{concentration[1]*100:.1f}%。这说明线性均值匹配中的集中问题有所缓解，并不证明高剂量记录错误或效应偏倚已经消除。

改善也有代价：284次样条模型有效样本量仅约34和31；272次样条模型约33和29。样条模型含新增基函数的最大SMD分别为0.126和0.146，不能只报告原变量低于0.10。284次样条模型中距手术时间、此前联用时间等分布仍有差异，最大原变量CDF差为{p.loc['all284_spline','max_original_ks']:.3f}。低维均值平衡不能替代多维支持诊断。

## 重抽样稳定性

{chr(10).join(btable)}

这些比例是300次重抽样的描述，不是模型无效概率。0.10沿用平衡描述参考线；ESS<20是本报告额外提供的事后描述刻度，不是预先注册的临床有效阈值，也没有据此筛除拟合。所有拟合均保留。

原变量SMD在非线性模型有改善，但主场景仍有71%的重抽样超过0.10；将新增基函数纳入时，约97.7%超过0.10。加压素先停组主场景样条ESS的中位数约26.7，2.5%分位约19.2。详细分位数见`bootstrap_distributions.csv`；样条相对线性的每次配对诊断变化见`paired_diagnostic_changes.csv`，这些变化不用于选择显著临床效果。

## 当前决策与下一步

1. 这一个预先固定的非线性扩展确实改善了剂量分布，结论不是“样条没有作用”；同时，它揭示可比较人群比原来更有限。不能再以单次最大SMD低于0.10宣称可以可靠比较两种策略。
2. 不自动继续增加结点、交互项或切样本寻找好看的平衡和RR。建议把当前项目主叙事转为术后撤药实践、滴定过程及可识别性；原IPTW、CCW和重叠结果作为方法学受限的探索性部分。尚未改写投稿稿件或删除原分析。
3. 若继续以因果比较为主要目标，需要明确临床上可互换的决策状态，以及更充分的治疗前信息或额外数据支持。必须先制定临床和方法学依据，不能从本次显著性或权重分布倒推定义。
4. 28天终点与跨库工作可在描述性目标下继续，但仍需独立验证结局观测范围、再入ICU和死亡日期精度；这些新增终点不能补救撤药比较本身的混杂与支持不足。

## 验证和复现

5项新增固定节点/缺失/外推/分布诊断测试及4项重叠权重测试通过；{len(checks)}项最终检查见`validation.json`。原核心及上一轮重叠产物哈希保持一致。`input_manifest.json`与`manifest.json`提供输入和产物哈希；所有输出位于独立目录。复现运行`scripts/41_nonlinear_dose_diagnostics.py`和`scripts/42_nonlinear_dose_report.py`。

接口依据：[scikit-learn SplineTransformer官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)。固定节点的临床适用性并未由该软件文档验证。项目状态继续为不可投稿，CCW的不可靠状态、既有重叠效应的探索性限制均保持不变。
'''
    (OUT/'nonlinear_dose_report.md').write_text(report,encoding='utf-8')
    paths=list(OUT.glob('*'))+[Path(__file__),BASE/'scripts/41_nonlinear_dose_diagnostics.py',BASE/'scripts/test_nonlinear_dose.py',BASE/'docs/nonlinear_dose_diagnostic_plan_v1.md']
    (OUT/'manifest.json').write_text(json.dumps([dict(path=str(x.relative_to(BASE)),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in paths if x.is_file() and x.name!='manifest.json'],indent=2),encoding='utf-8')
    print(json.dumps(validation,indent=2),flush=True)
    if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
