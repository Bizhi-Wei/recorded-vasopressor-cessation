"""Assemble audited tables, Chinese audit and an English descriptive manuscript."""
from pathlib import Path
import json
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
AUD=BASE/'rebuild_v3/outcome28_audit_v1'
OUT=BASE/'docs/manuscript_practice_v3'
OUT.mkdir(exist_ok=True)

def table(header,rows):
    return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def qfmt(s,d=1):
    q=s.dropna().quantile([.25,.5,.75])
    return f'{q.loc[.5]:.{d}f} ({q.loc[.25]:.{d}f}–{q.loc[.75]:.{d}f})'

def pct(n,den): return f'{int(n)}/{den} ({100*n/den:.1f}%)'

def main():
    d=pd.read_csv(BASE/'rebuild_v3/decision_analysis.csv')
    o=pd.read_csv(AUD/'observability_by_admission.csv')
    s=pd.read_csv(AUD/'summary.csv').set_index('group')
    v=json.loads((AUD/'validation.json').read_text())
    assert all(v['checks'].values())
    rows=[]; numerical=[]
    for var,label,digits,binary in [('age','Age in years',0,False),('male','Recorded male sex',0,True),('cabg','Coronary bypass code',0,True),('valve','Valve procedure code',0,True),('cpb','Cardiopulmonary bypass code',0,True),('weight','Recorded weight in kg',1,False),('hours_icu','Time since ICU entry in hours',1,False),('hours_surgery','Time since surgical date midnight in hours',1,False),('ne','Norepinephrine in mcg/kg/min',3,False),('va','Vasopressin in U/h',3,False),('map_mean','Prior six hour mean arterial pressure in mmHg',1,False),('lactate','Latest available lactate in mmol/L',1,False)]:
        r=[label]
        for a in [0,1]:
            x=d[d.A==a][var]; n=len(x); observed=int(x.notna().sum())
            text=pct(x.sum(),n) if binary else qfmt(x,digits)
            if observed<n: text+=f'; n={observed}'
            r.append(text)
            numerical.append(dict(table='Table1',variable=var,A=a,n=n,n_observed=observed,display=text))
        rows.append(r)
    t1='### Table 1 Characteristics at actual first single drug cessation\n\n'+table(['Characteristic','NE first n = 192','Vasopressin first n = 92'],rows)+'\n\nContinuous values are median (IQR); n marks available observations when incomplete. Surgery categories can overlap. The surgical-time variable begins at midnight on the procedure date, not the operative end. NE denotes norepinephrine.\n'
    w=pd.read_csv(BASE/'rebuild_v3/descriptive_practice_v1/pre_action_window_summary.csv')
    rows=[]
    for lo,hi in [(-6,-4),(-4,-2),(-2,0)]:
        for drug,label in [('ne','NE mcg/kg/min'),('va','Vasopressin U/h')]:
            r=[f'{lo} to {hi} h',label]
            for a in [0,1]:
                z=w[(w.A==a)&(w.window_start==lo)&(w.drug==drug)].iloc[0]
                r.append(f'{z["median"]:.3f} ({z.q25:.3f}–{z.q75:.3f})')
            rows.append(r)
    t2='### Table 2 Recorded doses in windows before cessation\n\n'+table(['Window','Drug and unit','NE first','Vasopressin first'],rows)+'\n\nValues are medians (IQR) of admission-level time-weighted means. Each window included all 192 and 92 admissions, respectively. Evaluable reconstruction assumes that an ICU interval without an active infusion row has zero dose.\n'
    rows=[]
    definitions=[('death28','Recorded death within 28 days'),('subsequent_icu_contact28','Subsequent same center ICU contact'),('index_icu_end_before28','Index ICU interval ended before day 28'),('any_unknown_active_rate','Any unknown active ICU infusion rate')]
    groups=['stable_all','action_ne_first','action_va_first']
    for var,label in definitions:
        rows.append([label]+[pct(s.loc[g,var],int(s.loc[g,'n'])) for g in groups])
    for var,label in [('recorded_icu_days28','Union of recorded ICU days'),('recorded_positive_pressor_days28','Recorded positive vasopressor days')]:
        rows.append([label]+[f'{s.loc[g,var+"_median"]:.2f} ({s.loc[g,var+"_q25"]:.2f}–{s.loc[g,var+"_q75"]:.2f})' for g in groups])
    t3='### Table 3 Audited records within 28 days\n\n'+table(['Measure','Eligibility n = 470','NE first n = 192','Vasopressin first n = 92'],rows)+'\n\nThe eligibility column begins at sustained low-dose co-infusion; the two cessation columns begin at actual cessation. They are overlapping populations with different anchors. ICU contact is limited to this center and includes planned or unplanned return. Days are medians (IQR) of recorded interval unions, not days alive or support-free days. Positive vasopressor duration is a partial ICU-record measure; unknown rates and ICU-external treatment are not imputed. No comparative inference is attached to this table.\n'
    diag=pd.read_csv(BASE/'rebuild_v3/nonlinear_dose_v1/point_diagnostics.csv').set_index('scenario')
    rows=[]
    for var,label in [('max_abs_smd','Maximum absolute SMD of original terms'),('ne_ks','NE weighted distribution difference'),('max_abs_smd_including_basis','Maximum absolute SMD including basis terms'),('ess_nefirst','Effective sample size NE first'),('ess_vasofirst','Effective sample size vasopressin first')]:
        rows.append([label]+[f'{diag.loc[g,var]:.3f}' for g in ['all284_linear','all284_spline']])
    t4='### Table 4 Dose model diagnostics in 284 cessation admissions\n\n'+table(['Diagnostic','Linear dose model','Nonlinear dose model'],rows)+'\n\nSMD denotes standardized mean difference with a fixed unweighted standard deviation denominator. Distribution difference is the maximum absolute difference between weighted empirical cumulative distributions. Effective sample size is a weight diagnostic, not a count of participants. These models do not establish clinical exchangeability.\n'
    net=pd.read_csv(BASE/'rebuild_v3/descriptive_practice_v1/two_drug_net_change_patterns.csv')
    patterns=net[['ne_net2h','va_net2h']].drop_duplicates()
    names={'approximately_flat':'Within 10%','net_down':'Reduced more than 10%','net_up':'Increased more than 10%'}
    rows=[]
    for r in patterns.itertuples():
        row=[names[r.ne_net2h],names[r.va_net2h]]
        for a,n in [(0,192),(1,92)]:
            z=net[(net.A==a)&(net.ne_net2h==r.ne_net2h)&(net.va_net2h==r.va_net2h)]
            row.append(pct(z.n.sum(),n))
        rows.append(row)
    st1='### Supplementary Table 1 Joint net changes during the final two hours\n\n'+table(['NE change','Vasopressin change','NE first','Vasopressin first'],rows)+'\n\nChanges compare the first and last segment rates and do not require monotonic change between them. Unlisted combinations were not observed.\n'
    st2='### Supplementary Table 2 Previously computed exploratory comparisons\n\n'+table(['Analysis','24 hour RR and 95% interval','Interpretation'],[
        ['Cessation IPTW 284 admissions','1.036 (0.737–1.346)','Residual imbalance and truncation sensitivity'],
        ['Cessation overlap 284 admissions','0.957 (0.730–1.275)','Post hoc association with distributional imbalance'],
        ['Cessation overlap 272 still low dose','1.079 (0.770–1.442)','Different selected population with unstable balance'],
        ['Eligibility clone censor weight 470 admissions','0.471 (0.346–0.636)','Unreliable strategy effect due to deadline misfit and poor support']])+'\n\nRR compares vasopressin-first with NE-first; intervals are the original patient-bootstrap percentile intervals. The eligibility-anchored composite differs from the cessation-anchored escalation endpoint. Values are retained for transparency, without asserting equivalent estimands or reliable causal effects.\n'
    tables='\n'.join([t1,t2,t3,t4]); supp='\n'.join([st1,st2])
    source=(BASE/'docs/manuscript_practice_v3_source.md').read_text(encoding='utf-8')
    manuscript=source.replace('{{TABLES}}',tables).replace('{{SUPPLEMENTARY_TABLES}}',supp)
    (OUT/'manuscript_practice_v3.md').write_text(manuscript,encoding='utf-8')
    (OUT/'tables.md').write_text(tables+'\n'+supp,encoding='utf-8')
    pd.DataFrame(numerical).to_csv(OUT/'table1_numeric_trace.csv',index=False)
    q=o[o.anchor=='action']; alive=q[q.death28==0]
    report=f'''# 28天结局可观测性审计

日期：2026-09-15。计算状态：{v['status']}。本报告支持限定范围的描述性结局，不验证撤药策略的因果效果。

## 结论与可报告范围

**28天死亡可以按数据库死亡链接规则作描述性报告；本院后续ICU接触可以作记录事件报告；完整28天无升压药、无机械通气及广义无器官支持天数目前不能成立。**

这三个结论来自不同的观察机制。官方3.1说明死亡链接至最后出院后一年；本轮以最后住院出院后365天作保守覆盖下限。470个稳定起点和284个实际停药起点均在覆盖内。没有死亡记录只表示在该链接系统中未记录死亡，不等于逐人主动随访或链接敏感性100%。[官方版本说明](https://physionet.org/content/mimiciv/3.1/)；[死亡字段](https://mimic.mit.edu/docs/iv/modules/hosp/patients.html)。

## 分母及结果

{table(['结局或记录','稳定起点470次','实际动作284次','NE先停192次','加压素先停92次'],[[label]+[pct(s.loc[g,var],int(s.loc[g,'n'])) for g in ['stable_all','action_all','action_ne_first','action_va_first']] for var,label in definitions])}

稳定队列为463名患者的470次住院；实际动作队列为283名患者的284次住院。两个起点不能混算，两队列不能相加。这些是住院级描述，未计算死亡/再入ICU的策略效应或P值。

稳定起点70次28天死亡中66次有医院时间戳、4次仅有日期；实际动作30次中27次有时间戳、3次仅有日期。所有死亡记录的时间顺序、跨表日期及边界检查均通过；无起点前死亡、无跨28天边界的日期不确定病例。死亡窗口为起点之后至28天截止时刻（含截止）；仅日期时以当日零点至次日零点的区间判定。完整逐住院分类在`observability_by_admission.csv`。

## 为什么旧的ICU自由天数不能直接沿用

按subject_id读取原始全部ICU stays，实际动作后28天有32次住院出现36次后续本院ICU接触；其中26次住院涉及本次hadm_id、6次涉及其他hadm_id。NE先停18/192，加压素先停14/92。这里只能说明同一数据库内的后续接触，不能区分计划返科和非计划再入，也不保证覆盖外院。[ICU stays定义](https://mimic.mit.edu/docs/iv/modules/icu/icustays.html)。

实际动作后，本次ICU区间时长中位数为5.00（IQR 2.66–10.32）天；合并所有后续ICU区间并去重叠后为5.28（2.92–11.30）天。在32次有后续接触的住院中，新增记录占用时长中位数为3.46（1.92–6.58）天。只用本次outtime会漏计这些记录。

26次住院的ICU行政区间延续至精确死亡时点之后，中位延后3.20小时、最大20.17小时。因此本报告保留的是**记录区间占用时间**，不是存活ICU时长；不以28减去该数冒充有效ICU-free days。后续ICU接触使用[起点,28天)窗口，不计恰在截止时刻新开始的区间。

## 为什么不能输出完整28天器官支持自由天数

实际动作队列254次未记录28天死亡的住院中，236次有ICU记录外时段；整体266/284次的本次ICU在28天前结束。30次死亡即使按某个自由天数定义赋0，也不能解决存活者的治疗历史缺失。全部窗口中ICU外时段还可能包括死亡后的时间，不能整体解释为活人失访时长。

重新从原始inputevents提取530个窗口相关ICU stay的六种升压药，覆盖重复ICU和其他住院；原470个index stay的精确剂量时间线全部复现。实际动作队列记录中的正剂量时间并集中位数1.40（0.42–3.89）天，仅为部分可见治疗记录；10次含未知活动速率，中位11.5分钟，最大5.55小时。未知与其他药物正剂量区间可能重合，两项不能相加成覆盖分解。

ICU内没有活动行按0只是重建假设；ICU外无行不能补成无支持。机械通气、RRT、MCS跨全部28天的时段本轮没有重建。旧脚本的VFD实际上来自最后升压药记录，并非通气测量；既不能继续称机械通气自由天数，也不能作为本轮有效升压药自由天数。[输注记录文档](https://mimic.mit.edu/docs/iv/modules/icu/inputevents.html)。

## 稿件处理

- 主体保留470次分母流转、284次实际动作、动作前滴定和剂量状态、短暂间断及可比性诊断。
- 28天死亡和后续本院ICU接触列为描述性次要记录；不以21/192与9/92推断等效或策略获益。
- 无器官支持天数不列为有效结果；旧的“没有临床结局差异”“稳健因果效应”“跨库可迁移性已证实”不沿用。
- 原探索性IPTW/重叠权重/CCW结果连同各自限制在补充材料保留，避免选择性呈现。

## 复现与核验

运行`scripts/45_outcome28_observability.py`、`scripts/46_outcome_report_manuscript_tables.py`。4项边界/区间并集单元测试通过；{len(v['checks'])}项原始链接与一致性检查通过。`source_manifest.json`包含四张原始表的完整SHA256和两个队列输入哈希；`validation.json`记录每项检查。既有核心、描述及非线性诊断清单中的文件哈希保持一致。

这些检查只验证所列计算和记录一致性，没有验证外院捕获、死亡链接漏报率、医生意图或撤药因果效应。原投稿Word/PDF及其提交门禁保持原状，新稿为独立工作版本。
'''
    (AUD/'outcome28_observability_report.md').write_text(report,encoding='utf-8')
    ledger=[dict(endpoint='28-day recorded mortality',status='DESCRIPTIVE_WITHIN_DOCUMENTED_LINKAGE_WINDOW',limitation='Linkage sensitivity not independently verified'),
        dict(endpoint='Subsequent same-center ICU contact',status='OBSERVED_SAME_CENTER_EVENTS_ONLY',limitation='Planned returns and external care not resolved'),
        dict(endpoint='Recorded ICU interval union',status='ADMINISTRATIVE_RECORD_DURATION_ONLY',limitation='Not death adjusted; includes post-death administrative intervals'),
        dict(endpoint='Recorded positive vasopressor interval union',status='PARTIAL_ICU_RECORD_EXPOSURE_ONLY',limitation='Unknown rates and ICU-external care not imputed'),
        dict(endpoint='28-day vasopressor-free days',status='NOT_ESTABLISHED',limitation='Incomplete longitudinal exposure'),
        dict(endpoint='28-day ventilator-free days',status='NOT_REBUILT',limitation='Legacy pressor proxy invalid as ventilation measure'),
        dict(endpoint='28-day broad organ-support-free days',status='NOT_REBUILT',limitation='Longitudinal ventilation RRT MCS not reconstructed')]
    pd.DataFrame(ledger).to_csv(AUD/'endpoint_evidence_status.csv',index=False)
    print('Created audit, manuscript and six tables',flush=True)

if __name__=='__main__': main()
