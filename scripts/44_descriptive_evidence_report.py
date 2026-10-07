"""Integrated descriptive evidence with provenance and retained causal limitations."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1];SRC=BASE/'rebuild_v3';OUT=SRC/'descriptive_practice_v1'

def verify(path):
    for r in json.loads(path.read_text(encoding='utf-8')):
        p=Path(r['path']);p=p if p.is_absolute() else BASE/p
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:return False
    return True

def main():
    summary=json.loads((OUT/'summary.json').read_text());states=pd.read_csv(OUT/'practice_state_counts.csv').set_index('practice_state')
    doses=pd.read_csv(OUT/'decision_practice_summary.csv').set_index(['A','variable'])
    windows=pd.read_csv(OUT/'pre_action_window_summary.csv').set_index(['A','window_start','drug'])
    net=pd.read_csv(OUT/'two_drug_net_change_patterns.csv')
    measurements=pd.read_csv(SRC/'design_feasibility/measurement_summary.csv').set_index('arm')
    primary=pd.read_csv(SRC/'decision_results.csv').set_index('estimand')
    overlap=pd.read_csv(SRC/'overlap_v1/results.csv').set_index(['scenario','estimand'])
    ccw=pd.read_csv(SRC/'ccw_results.csv').set_index('estimand')
    def count(arm,ne,va):
        x=net[(net.A==arm)&(net.ne_net2h==ne)&(net.va_net2h==va)]
        return int(x.n.sum())
    def med(arm,var,decimals=3):
        r=doses.loc[(arm,var)];return f'{r["median"]:.{decimals}f}（{r.q25:.{decimals}f}–{r.q75:.{decimals}f}）'
    def eff(row):return f'{row.estimate:.3f}（{row.lo:.3f}–{row.hi:.3f}）'
    evidence=[
        dict(analysis='实际撤药IPTW，284次',endpoint='动作后24h本次ICU反跳',rr=eff(primary.loc['iptw_relapse24_rr']),status='探索性校正关联；剂量残余失衡及权重截尾敏感',source='rebuild_v3/core_rebuild_report.md'),
        dict(analysis='实际撤药重叠权重，284次',endpoint='动作后24h本次ICU反跳',rr=eff(overlap.loc[('all284','relapse24_rr')]),status='事后探索性关联；剂量分布及重抽样平衡不足',source='rebuild_v3/overlap_v1/overlap_report.md'),
        dict(analysis='仍稳定者重叠权重，272次',endpoint='动作后24h本次ICU反跳',rr=eff(overlap.loc[('stable272','relapse24_rr')]),status='不同人群的事后敏感性；支持及平衡仍有限',source='rebuild_v3/overlap_v1/overlap_report.md'),
        dict(analysis='稳定状态CCW，470次',endpoint='首次稳定后24h支持升级或死亡',rr=eff(ccw.loc['ccw_rr']),status='不能作为可靠策略效应；截止点模型失配、灵活模型ESS极低',source='rebuild_v3/weight_time_audit/audit_report.md')]
    pd.DataFrame(evidence).to_csv(OUT/'evidence_status_ledger.csv',index=False)
    sources=[SRC/'design_feasibility/measurement_summary.csv',SRC/'decision_results.csv',SRC/'overlap_v1/results.csv',SRC/'ccw_results.csv',SRC/'nonlinear_dose_v1/validation.json']
    (OUT/'report_input_manifest.json').write_text(json.dumps([dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],indent=2),encoding='utf-8')
    proc=subprocess.run([sys.executable,str(BASE/'scripts/test_descriptive_practice.py')],capture_output=True,text=True)
    (OUT/'unit_test_output.txt').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    checks=summary['checks'].copy();checks.update(unit_tests_pass=proc.returncode==0,
        build_inputs_unchanged=verify(OUT/'input_manifest.json'),report_inputs_unchanged=verify(OUT/'report_input_manifest.json'),
        prior_core_unchanged=verify(SRC/'core_manifest.json'),prior_overlap_unchanged=verify(SRC/'overlap_v1/manifest.json'),
        prior_nonlinear_diagnostics_unchanged=verify(SRC/'nonlinear_dose_v1/manifest.json'),
        windows_complete_for_report=bool(windows.n_group.eq(windows.n_full).all()))
    validation=dict(status='DESCRIPTIVE_PRACTICE_CHECKS_PASS' if all(checks.values()) else 'DESCRIPTIVE_PRACTICE_CHECKS_FAILED',checks=checks,
        research_focus='Recorded withdrawal practice, titration trajectories and limits of comparability',
        new_effects_computed=False,causal_assumptions_verified=False,submission_ready=False)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    st=['| 稳定后12小时内的记录状态 | 住院次数 | 占470次稳定池 |','|---|---|---|']
    for key,label in [('norepinephrine','NE首次先停'),('vasopressin','加压素首次先停'),('simultaneous','两药同刻停止'),('no_action_full12h','完整12小时可观测，未记录首次动作'),('no_action_observation_ended','未记录动作，观察提前结束'),('no_action_unknown_dose','未记录动作，期间出现未知剂量')]:
        r=states.loc[key];st.append(f'| {label} | {int(r.n_admissions)} | {r.fraction_pool*100:.1f}% |')
    wt=['| 动作前窗口 | NE先停组：NE | NE先停组：加压素 | 加压素先停组：NE | 加压素先停组：加压素 |','|---|---|---|---|---|']
    for low in [-6,-4,-2]:
        vals=[]
        for arm,drug in [(0,'ne'),(0,'va'),(1,'ne'),(1,'va')]:
            r=windows.loc[(arm,low,drug)];vals.append(f'{r["median"]:.3f}（{r.q25:.3f}–{r.q75:.3f}）')
        wt.append('| '+f'{low}至{low+2}小时'+' | '+' | '.join(vals)+' |')
    et=['| 已有分析 | 结局起点与定义 | RR及原95%区间 | 当前证据等级 |','|---|---|---|---|']
    for e in evidence:et.append(f"| {e['analysis']} | {e['endpoint']} | {e['rr']} | {e['status']} |")
    dominant=count(0,'net_down','approximately_flat')
    report=f'''# 心脏术后双升压药撤除：实践、滴定过程与可比性限制

日期：2026-09-15。**当前研究定位：描述记录中的撤药实践及其治疗前差异，保留比较效应为有明确限制的探索分析。** 本报告整合已完成证据并新增动作前滴定描述，不是投稿定稿。状态：{validation['status']}；项目仍不可投稿。

## 当前能支持的主要发现

在严格手术日期口径的470次稳定联用住院中，12小时内首次NE先停192次、加压素先停92次。NE先停者在动作前两小时最常见模式为NE净下调、加压素近似持平（{dominant}/192，{dominant/192*100:.1f}%）；加压素先停组模式较分散。两类观察到的撤药顺序伴随此前不同的剂量状态，不能把最终停止哪种药等同于在相同状态下随机选择顺序。

这些是记录分布与净变化描述，不证明医师选择的原因、既定科室流程、策略优劣或临床获益。记录结束也不必然代表有意完成临床撤药。

## 人群和观察范围

严格手术日期之后、连续两小时NE+加压素低剂量联用的首次稳定状态：470次住院、463名患者。12小时内首次单药撤除：284次住院、283名患者；其中NE先停192次（191名患者），加压素先停92次（92名患者）。一位患者可有多次住院，分组患者数不应与总患者数机械相加。仅有手术日期、排除手术当日进入者的定义使其不能代表全部即刻心脏术后人群。

{chr(10).join(st)}

“没有动作记录”不是“明确选择不撤药”。其中181次在重建规则下可覆盖完整12小时，另2次分别因观察结束及未知剂量不能作完整窗口判断。无输注行按原重建规则视为无输注，完整覆盖仍不意味着病历绝无漏记。

首次稳定至动作的时间中位数（IQR）：NE先停{med(0,'delay_h',2)}小时，加压素先停{med(1,'delay_h',2)}小时。此统计仅在发生首次单药动作的284次住院中计算，不是全稳定池的撤药等待时间分布。

## 动作前剂量状态和轨迹

动作前左极限NE剂量中位数（IQR）：NE先停{med(0,'ne_action')}、加压素先停{med(1,'ne_action')} mcg/kg/min；加压素剂量分别为{med(0,'va_action')}、{med(1,'va_action')} U/h。

下表先在每次住院的两小时窗口内按精确输注区段做时间加权均值，再跨住院报告中位数（IQR）。三个窗口每组所有住院在本轮重建规则下均有完整、已知剂量覆盖。不是把多条输注记录当作独立患者，也不是对窗口首末值简单平均。NE单位mcg/kg/min，加压素单位U/h。

{chr(10).join(wt)}

观察到的NE先停组在三个窗口的NE时间加权均值中位数依次降低，而加压素中位数维持约2.4 U/h；加压素先停组两药的窗口均值中位数均下降。跨窗口组中位数变化不能证明每位患者都单调减量。

完整动作前两小时的首末段净变化以相对10%分档，该值只是描述阈值，不是临床显著性标准。NE先停组NE净下调/加压素近似持平{dominant}次、两药均净下调{count(0,'net_down','net_down')}次；加压素先停组分别为{count(1,'net_down','approximately_flat')}次和{count(1,'net_down','net_down')}次，另有NE近似持平/加压素净下调{count(1,'approximately_flat','net_down')}次。完整组合及分母见`two_drug_net_change_patterns.csv`。净首末变化无法排除中间先加量再减量。

## 短暂停顿和动作时状态

已有测量复核显示15分钟内重启被撤药物：NE先停{int(measurements.loc[0,'restart_15min_n'])}/192，加压素先停{int(measurements.loc[1,'restart_15min_n'])}/92；两小时内分别为{int(measurements.loc[0,'restart_120min_n'])}和{int(measurements.loc[1,'restart_120min_n'])}次。此为记录重启描述，不自动等同于临床反跳；不以未来是否持续停药删样本。

动作前仍符合连续两小时低剂量稳定者为185/192和87/92，因此稳定池达到条件后，实际动作时并非所有住院仍符合相同状态。47次源记录时序已核对，临床意图仍未判明；异常高NE段原始表速率与输注量/时长算术一致，但真实性不能由这些可能共享来源的字段独立验证。详见[记录设计复核](../design_feasibility/next_stage_recommendation.md)和[原始剂量核对](../source_rate_audit/source_rate_findings.md)。

## 既有比较效应的正确位置

{chr(10).join(et)}

上述RR方向均为加压素先停相对NE先停，数字直接读取既有结果文件，本轮未重拟合比较模型。两个时间起点及复合结局不同，不得把RR接近或不同当作验证或反驳。区间跨越1也不代表等效；没有证据足以建议优先撤哪一种药。

后续非线性剂量诊断改善NE分布但压缩支持：主场景两臂ESS约34和31，仍有71%的重抽样原变量最大SMD>0.10，含新增基函数时约97.7%。这支持保留“治疗状态不充分可比”的限制，而不是以模型更复杂将效应升级。见[非线性诊断](../nonlinear_dose_v1/nonlinear_dose_report.md)。

## 当前可用于项目讨论的研究问题

核心问题可表述为：在手术日期之后仍持续接受低剂量NE与加压素联用的患者中，记录中的首次单药撤除如何发生，其此前滴定轨迹和决策状态有何差异，这些差异给观察性顺序比较带来哪些限制？

主体证据应包括稳定池到动作的分母、两药动作前轨迹、短间断的测量语义和共同支持诊断。原比较效应作为探索部分完整保留。这里没有证明此主题的新颖性、外部普遍性或投稿可接受性；投稿论证与临床解释仍需后续工作。

## 未完成内容与后续顺序

1. 按这个研究定位核对剩余变量和临床表述，将稿件的策略优劣/稳健因果主张改为上述记录实践及可识别性问题；旧Word/PDF仍为历史产物，本轮未生成投稿稿件。
2. 对28天死亡、再入ICU及支持时长做可观测性审计，再决定能报告哪些描述性结局。严禁把出ICU后的无记录全部补成无升压药或无机械通气。
3. 外部数据库先做同口径队列和记录语义核对，分别报告可用和不可用信息；跨库实践差异不自动是同一因果效应的验证。

## 复现和验证

新增分析固定于[描述计划](../../docs/descriptive_practice_plan_v1.md)。执行`scripts/43_descriptive_practice.py`与`scripts/44_descriptive_evidence_report.py`；逐住院摘要、精确分段和汇总CSV均在本目录。4项时间积分/缺失/净变化测试及{len(checks)}项最终检查通过，原核心、重叠及非线性文件哈希保持一致。`validation.json`仅标记计算及所列核对通过，未验证医师意图或因果假设。产物哈希见`manifest.json`。
'''
    (OUT/'practice_evidence_report.md').write_text(report,encoding='utf-8')
    paths=list(OUT.glob('*'))+[Path(__file__),BASE/'scripts/43_descriptive_practice.py',BASE/'scripts/test_descriptive_practice.py',BASE/'docs/descriptive_practice_plan_v1.md']
    (OUT/'manifest.json').write_text(json.dumps([dict(path=str(p.relative_to(BASE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths if p.is_file() and p.name!='manifest.json'],indent=2),encoding='utf-8')
    print(json.dumps(validation,indent=2),flush=True)
    if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
