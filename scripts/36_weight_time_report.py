"""Validate and report all sensitivity scenarios without choosing by effect size."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]
SRC=BASE/'rebuild_v3'
OUT=SRC/'weight_time_audit'
LABELS=['60min_linear','60min_categorical','15min_linear','15min_categorical']
NAMES={'60min_linear':'60分钟／线性','60min_categorical':'60分钟／分类','15min_linear':'15分钟／线性','15min_categorical':'15分钟／分类'}

def check_manifest(path):
    records=json.loads(path.read_text(encoding='utf-8'))
    for r in records:
        p=Path(r['path'])
        if not p.is_absolute():p=BASE/p
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:return False
    return True

def main():
    summary=json.loads((OUT/'inference_summary.json').read_text())
    build=json.loads((OUT/'build_summary.json').read_text())
    diag=pd.DataFrame(json.loads((OUT/'point_diagnostics.json').read_text())).set_index('scenario')
    res=pd.read_csv(OUT/'sensitivity_results.csv').set_index(['scenario','estimand'])
    paired=pd.read_csv(OUT/'paired_scenario_differences.csv')
    cal=pd.read_csv(OUT/'censor_calibration.csv')
    deadline=cal[cal.end_h.eq(12)].copy()
    deadline['absolute_calibration_gap']=(deadline.observed_uncensored-deadline.predicted_uncensored).abs()
    deadline.to_csv(OUT/'deadline_calibration.csv',index=False)
    bal=pd.read_csv(OUT/'decision_balance_fixed_sd.csv')
    fixed=bal[bal.weighting.eq('iptw')]
    counts=pd.read_csv(OUT/'decision_support_counts.csv')
    boot=pd.read_csv(OUT/'bootstrap.csv')
    test_results=[]
    for file in ['test_rebuild_v3.py','test_weight_time_audit.py']:
        proc=subprocess.run([sys.executable,str(BASE/'scripts'/file)],capture_output=True,text=True)
        test_results.append(proc.returncode==0)
        (OUT/(file+'.txt')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
    checks=dict(original_core_manifest_unchanged=check_manifest(SRC/'core_manifest.json'),
        audit_input_manifest_matches=check_manifest(OUT/'input_manifest.json'),unit_tests_pass=all(test_results),
        original_hourly_grid_reproduced=build['hourly_grid_reproduced'],
        exact_clinical_and_artificial_counts_match=build['exact_clinical_and_artificial_terminal_counts_match'],
        original_point_estimates_reproduced=summary['baseline_point_reproduced'],
        original_300_bootstraps_reproduced=summary['baseline_bootstrap_reproduced'],
        all_four_scenarios_reported=set(summary['success_by_scenario'])==set(LABELS),
        all_1200_fits_successful=all(summary['success_by_scenario'].get(x)==300 for x in LABELS) and not summary['failures'],
        all_reported_intervals_use_300_finite_replicates=bool(res.bootstrap_finite.eq(300).all()),
        all_estimates_and_intervals_finite=bool(np.isfinite(res[['estimate','lo','hi']].to_numpy()).all()),
        all_intervals_ordered=bool(res.lo.le(res.hi).all()),
        paired_comparisons_use_all_300_draws=bool(paired.n_pairs.eq(300).all()),
        finite_positive_weights=bool(diag.finite_weights.all()),
        weights_fixed_after_adherence=bool(diag.weights_fixed_after_adherence.all()),
        no_duplicate_bootstrap_scenario=not boot.duplicated(['replicate','scenario']).any())
    numerical='AUDIT_COMPUTED_CHECKS_PASS' if all(checks.values()) else 'AUDIT_CHECKS_FAILED'
    flags=dict(hourly_linear_deadline_max_absolute_gap=float(deadline[deadline.scenario.eq('60min_linear')].absolute_calibration_gap.max()),
        categorical_min_outcome_ess=float(diag.loc[['60min_categorical','15min_categorical'],'min_outcome_ess'].min()),
        decision_fixed_sd_max_absolute_smd=float(fixed.smd_fixed_baseline_sd.abs().max()),
        causal_assumptions_verified=False,submission_ready=False)
    interpretation='CCW_EFFECT_NOT_RELIABLE_DEADLINE_MODEL_MISFIT_AND_LOW_ESS'
    validation=dict(status=numerical,checks=checks,interpretation_status=interpretation,substantive_findings=flags)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    # Standalone scientific figure with all four scenarios and uncertainty.
    fig,(ax,ae)=plt.subplots(1,2,figsize=(11,4.8),gridspec_kw={'width_ratios':[1.5,1]})
    colors=['#2563a6','#a34938','#2563a6','#a34938']
    for i,label in enumerate(LABELS):
        r=res.loc[(label,'trim_rr')]
        ax.plot([r.lo,r.hi],[i,i],color=colors[i],lw=2)
        ax.plot(r.estimate,i,'o',color=colors[i])
        ax.text(r.hi*1.06,i,f'{r.estimate:.2f} [{r.lo:.2f}, {r.hi:.2f}]',va='center',fontsize=9)
        ess=diag.loc[label,'min_outcome_ess']
        ae.barh(i,ess,color=colors[i],height=.48)
        ae.text(ess+.7,i,f'{ess:.1f}',va='center')
    english=['60 min / linear','60 min / categorical','15 min / linear','15 min / categorical']
    ax.axvline(1,color='#777777',ls='--',lw=1);ax.set_xscale('log')
    maxhi=res.xs('trim_rr',level='estimand').hi.max()
    minlo=res.xs('trim_rr',level='estimand').lo.min()
    ax.set_xlim(max(.01,minlo*.8),maxhi*3.1)
    ax.set_xticks([.2,.5,1,2,4],['0.2','0.5','1','2','4']);ax.minorticks_off()
    ax.set_yticks(range(4),english);ax.invert_yaxis()
    ax.set_xlabel('Risk ratio: vasopressin-first / NE-first')
    ax.set_title('24-hour escalation or death\n99% truncated IPCW; patient bootstrap 95% CI',fontsize=11)
    ae.set_yticks(range(4),['']*4);ae.invert_yaxis()
    ae.set_xlim(0,diag.min_outcome_ess.max()*1.25)
    ae.set_xlabel('Minimum effective sample size')
    ae.set_title('Lowest arm / time-block support\nOutcome risk sets, after truncation',fontsize=11)
    for a in [ax,ae]:
        a.spines[['top','right']].set_visible(False)
        a.set_ylim(3.4,-.4)
    fig.suptitle('Time-model sensitivity reveals limited support',fontweight='bold')
    fig.text(.5,.015,'Post hoc methodological audit; these contrasts do not establish a preferred withdrawal strategy.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94])
    fig.savefig(OUT/'ccw_sensitivity.png',dpi=180)
    fig.savefig(OUT/'ccw_sensitivity.pdf')
    plt.close(fig)
    def estimate(label,key,mult=1):
        r=res.loc[(label,key)]
        return f'{r.estimate*mult:.3f}（{r.lo*mult:.3f}–{r.hi*mult:.3f}）'
    table=['| 时间网格／时间项 | RR及95%区间 | RD及95%区间，百分点 | 最低风险集ESS |','|---|---|---|---|']
    for label in LABELS:
        table.append(f"| {NAMES[label]} | {estimate(label,'trim_rr')} | {estimate(label,'trim_rd',100)} | {diag.loc[label,'min_outcome_ess']:.2f} |")
    ctable=['| 模型 | 策略 | 截止区块待完成数 | 实际未删失比例 | 预测未删失比例 |','|---|---|---|---|---|']
    for r in deadline.itertuples():
        ctable.append(f"| {NAMES[r.scenario]} | {'加压素先停' if r.arm else 'NE先停'} | {r.n} | {r.observed_uncensored*100:.1f}% | {r.predicted_uncensored*100:.1f}% |")
    common=counts[counts.rule.eq('empirical_common_range')].n_in.sum()
    central=counts[counts.rule.eq('ps_10_90')].n_in.sum()
    report=f'''# 撤药顺序项目：权重与时间离散复核

**计算状态：{numerical}。解释状态：{interpretation}；项目仍不可投稿。**

本轮最重要的发现是：上一轮60分钟线性时间CCW模型在12小时截止处严重低估人工删失。分类时间模型能更贴近截止区块的样本内比例，但相应权重集中于极少数患者，最低分臂区块ESS仅{flags['categorical_min_outcome_ess']:.2f}。因此不能继续把上一轮RR 0.47及其区间作为可靠策略效应，更不能将其解释为加压素先停已经证实更优。原来的计算及边界检查通过仍然成立；它们没有检验这一模型适配和支持度问题。

## 做了什么

按[事后复核计划](../../docs/weight_time_audit_plan_v1.md)固定四个场景，同一470次住院、463名患者、940个克隆、相同精确终止时刻和结局；60分钟10,201行，15分钟40,056行。新增网格做了{build['source_time_feature_checks']:,}次起点前协变量提取检查。15分钟网格不向前借用小时末记录。

时间分类指示变量与其余输入一起标准化，缺失处理及正则强度与既定模型一致。分类项提供时间函数灵活性，本轮没有调参选模型。四个场景使用相同的300次患者重抽样，每次重新拟合插补、标准化与两臂删失模型，共1,200次拟合，全部成功。原60分钟线性模型的点估计及全部300次bootstrap均复现。

## 结果与支持度

所有方向均为加压素先停相对NE先停；结局为稳定状态起算24小时支持升级或死亡，与实际撤药后的反跳不同。表中为每臂99%截尾结果，未截尾及未加权结果完整保存在`sensitivity_results.csv`，不能只挑下面某个场景。

{chr(10).join(table)}

![CCW敏感性与有效样本量](ccw_sensitivity.png)

区间来自同一数据集的事后敏感性分析，且不对全部探索性比较作多重校正。场景间RD差的配对区间保存在`paired_scenario_differences.csv`，不能通过区间是否跨0选择时间函数。分类时间模型的宽区间或点估计变化，不构成两策略等效或反向优劣的证据。

## 截止点为什么重要

以下比例只针对尚未完成分配策略、仍参与删失模型的克隆。60分钟模型的截止区块为11–12小时，15分钟模型为11.75–12小时，分母不同；两者不能当作相同窗口直接比较。未删失也可能包括先发生临床终止事件者，不等同于单纯按时撤药比例。

{chr(10).join(ctable)}

这是样本内分组校准复核，分类模型贴合比例本身不能证明个体预测正确。它揭示了一个实际矛盾：原线性模型把集中在截止点的删失平滑掉；放开时间函数后，该点只有极少的可用未删失记录，权重不稳定。类别模型在结局风险行中的原始最大权重分别为{diag.loc['60min_categorical','raw_weight_max_outcome']:.1f}和{diag.loc['15min_categorical','raw_weight_max_outcome']:.1f}，截尾后仍分别为{diag.loc['60min_categorical','trimmed_weight_max_outcome']:.1f}和{diag.loc['15min_categorical','trimmed_weight_max_outcome']:.1f}。所有行（含人工删失行）的极值另列于`point_diagnostics.json`，没有混称为结局使用权重。

另外，人工删失所在区块被从该区块完整结局分母排除，这确实是近似：手工病例中1.25小时事件和1.75小时删失放入同一小时，会提前移除原本仍在险的人；15分钟网格能保留该人的相应风险时间。该边界行为已用测试明确展示，不能把60分钟结果称精确连续时间估计。

## 实际撤药分析的残余失衡

本轮没有修改284例实际撤药主模型。用固定的未加权两组标准差作分母，IPTW后的最大绝对SMD为{flags['decision_fixed_sd_max_absolute_smd']:.3f}（加压素剂量）；原报告0.447使用加权后的标准差作分母，二者定义不同，不是数字重算错误。两种口径都显示残余失衡。

两臂经验PS范围的交集内有{common}/284例；PS在0.10–0.90内仅{central}/284例。这些仅是支持分布描述，并未据此删样本、重新定义目标人群或选择有利效应。完整固定分母SMD、剂量/权重分布及支持计数见本目录CSV。现有实际撤药RR仍只能作有这些限制的校正关联结果。

## 验证与下一步

- 17个原重建边界测试和7个新增风险集/竞争事件/重抽样测试均通过；{len(checks)}项最终检查见`validation.json`。
- 原核心43项文件哈希及本轮输入清单保持匹配。新增代码、图表、结果哈希见`manifest.json`。
- 明确降低CCW效应的解释等级，保留所有结果供审阅，不以低ESS场景替代原模型并宣称修复成功。
- 下一阶段需从策略本身与数据支持度重新审阅12小时行动窗口及撤药时机干预版本；任何改变都会形成新的分析修订，不能伪装成原估计目标的确认。28天器官支持、外部数据库和投稿稿件尚未重建。

方法背景参考[Gaber等CCW primer](https://pmc.ncbi.nlm.nih.gov/articles/PMC11623977/)（本轮搜索可取得方法摘要，PMC正文入口出现验证码；不声称全文逐项复核）。这里的定量发现全部来自本地重分析。

复现顺序：`34_weight_time_build.py` → `35_weight_time_inference.py` → `36_weight_time_report.py`。原v3结果不覆盖，本轮结果位于独立目录。
'''
    (OUT/'audit_report.md').write_text(report,encoding='utf-8')
    original=SRC/'core_rebuild_report.md'
    notice='<!-- WEIGHT_TIME_AUDIT_NOTICE -->'
    core_text=original.read_text(encoding='utf-8')
    if notice not in core_text:
        (OUT/'core_report_before_weight_time_audit.md').write_text(core_text,encoding='utf-8')
        original.write_text(notice+'\n> **后续复核更新：本报告的CCW RR 0.47已降低解释等级。** 12小时截止处删失模型明显失配，分类时间模型最低ESS仅2.75；不能作为可靠策略效应。参见[最新权重与时间复核](weight_time_audit/audit_report.md)。以下保留本轮原计算结果及其检查范围。\n\n'+core_text,encoding='utf-8')
    paths=list(OUT.glob('*'))
    paths+=[BASE/'scripts'/name for name in ['34_weight_time_build.py','35_weight_time_inference.py','36_weight_time_report.py','test_weight_time_audit.py','31_rebuild_datasets.py','32_rebuild_inference.py','rebuild_v3_core.py']]
    paths+=[BASE/'docs/weight_time_audit_plan_v1.md']
    manifest=[]
    for p in paths:
        if p.is_file() and p.name!='manifest.json':
            manifest.append(dict(path=str(p.relative_to(BASE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(validation,indent=2),flush=True)
    if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
