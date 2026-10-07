"""Run the locked measurement sensitivities without clinical outcome fitting."""
from pathlib import Path
import hashlib
import importlib.util
import json
import pickle
import subprocess
import sys
import numpy as np
import pandas as pd
from rebuild_v3_core import seconds, at, first_unknown, build_timeline, NE, VA
from cessation_sensitivity_core import candidates, prepare_rows, ending_source, bridge_gaps

BASE=Path(__file__).resolve().parents[1]; SRC=BASE/'rebuild_v3'; OUT=SRC/'cessation_sensitivity_v1'
SCENARIOS=['reference','paused_excluded','paused_reclassified','gap_5min','gap_15min','gap_30min']
spec=importlib.util.spec_from_file_location('practice',BASE/'scripts/43_descriptive_practice.py')
practice=importlib.util.module_from_spec(spec);spec.loader.exec_module(practice)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(path,obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True)
    inputs=[SRC/'stable_pool.csv',SRC/'timelines.pkl',SRC/'source_pressors.pkl',
            BASE/'docs/cessation_sensitivity_plan_v1.md',BASE/'scripts/rebuild_v3_core.py',
            BASE/'scripts/43_descriptive_practice.py',BASE/'scripts/cessation_sensitivity_core.py',Path(__file__)]
    write_json(OUT/'input_manifest.json',[dict(path=str(p.relative_to(BASE)),sha256=sha(p)) for p in inputs])
    pool=pd.read_csv(SRC/'stable_pool.csv')
    for c in ['stable_t0','action_time','intime','outtime','deathtime']:pool[c]=pd.to_datetime(pool[c])
    timelines=pickle.load((SRC/'timelines.pkl').open('rb')); raw=pd.read_pickle(SRC/'source_pressors.pkl')
    bystay=dict(tuple(raw.groupby('stay_id',sort=False)))
    individual=[]; evidence=[]; gaps=[]; terminal=[]; descriptions=[]
    reproduced=[]; refmatched=[]; doseunchanged=[]
    for r in pool.itertuples():
        tl=timelines[r.stay_id]; original=tl['rates'].copy(); start=seconds(r.stable_t0); deadline=start+43200
        obs_end=min(seconds(r.outtime),seconds(r.deathtime) if pd.notna(r.deathtime) else np.inf)
        end=min(deadline,obs_end); unknown=first_unknown(tl,start,end)
        rebuilt=build_timeline(bystay[r.stay_id],r.intime,r.outtime)
        reproduced.append(np.array_equal(rebuilt['bounds'],tl['bounds']) and np.array_equal(rebuilt['rates'],original,equal_nan=True))
        prepared=prepare_rows(bystay[r.stay_id]); source_cache={}
        def audit_event(event,scenario):
            t,cls,stopped=event; flags=[]
            for j in stopped:
                key=(t,int(j))
                if key not in source_cache:
                    source_cache[key]=ending_source(prepared,[NE,VA][j],t,at(tl,t,True)[j])
                e=source_cache[key]; flags.append(e['paused'])
                evidence.append(dict(stay_id=r.stay_id,scenario=scenario,action_seconds=t,classification=cls,**e))
            return any(flags)
        events=list(candidates(tl,start,end))
        for event in events: audit_event(event,'reference_candidates')
        baseline=events[0] if events else (np.inf,'none',[])
        expected=seconds(r.action_time) if pd.notna(r.action_time) else np.inf
        refmatched.append(baseline[0]==expected and baseline[1]==r.first_drug)
        ispaused=bool(events and audit_event(baseline,'reference_selected'))
        choices={'reference':baseline,'paused_excluded':(np.inf,'paused_unclassified',[]) if ispaused else baseline}
        accepted=[e for e in events if not audit_event(e,'paused_reclassification_candidates')]
        choices['paused_reclassified']=accepted[0] if accepted else (np.inf,'no_accepted_action' if events else 'none',[])
        for minutes in [5,15,30]:
            scenario=f'gap_{minutes}min'
            virtual,g,u=bridge_gaps(tl,minutes,start,deadline,obs_end)
            gaps.extend(dict(stay_id=r.stay_id,scenario=scenario,**v) for v in g)
            terminal.extend(dict(stay_id=r.stay_id,scenario=scenario,**v) for v in u)
            ev=list(candidates(virtual,start,end));choices[scenario]=ev[0] if ev else (np.inf,'none',[])
            if ev: audit_event(ev[0],scenario+'_selected')
        for scenario,(t,cls,stopped) in choices.items():
            finite=np.isfinite(t); basefinite=np.isfinite(baseline[0]); sameclass=cls==baseline[1]
            sameevent=sameclass and ((not finite and not basefinite) or (finite and t==baseline[0]))
            row=dict(scenario=scenario,stay_id=r.stay_id,hadm_id=r.hadm_id,subject_id=r.subject_id,
                stable_t0=str(r.stable_t0),reference_class=baseline[1],reference_action_seconds=baseline[0],
                reference_paused=ispaused,classification=cls,action_seconds=t,
                action_time=str(pd.Timestamp(t,unit='s')) if finite else '',
                same_class=sameclass,same_event=sameevent,
                delay_from_reference_minutes=(t-baseline[0])/60 if finite and basefinite else np.nan,
                reference_single=baseline[1] in [NE,VA],selected_single=cls in [NE,VA],
                direct_order_switch=baseline[1] in [NE,VA] and cls in [NE,VA] and cls!=baseline[1],
                original_window_complete=obs_end>=deadline and unknown>=end,
                window_status='unknown_active_dose' if unknown<end else ('observation_ended' if obs_end<deadline else 'full12h'),
                actual_both_positive_before=bool((at(tl,t,True)[:2]>0).all()) if finite else False)
            individual.append(row)
            if cls in [NE,VA]:
                pre=at(tl,t,True)
                d=dict(scenario=scenario,stay_id=r.stay_id,classification=cls,delay_h=(t-start)/3600,
                       ne_action=pre[0],va_action=pre[1],actual_both_positive_before=row['actual_both_positive_before'])
                for j,name in [(0,'ne'),(1,'va')]:
                    w=practice.summarize_window(tl,t-7200,t,j)
                    d[name+'_net2h']=practice.net_category(w['start_rate'],w['end_rate'],w['full_window'])
                    d[name+'_window_complete']=w['full_window']
                    d[name+'_mean2h']=w['mean']
                descriptions.append(d)
        doseunchanged.append(np.array_equal(original,tl['rates'],equal_nan=True))
    ind=pd.DataFrame(individual); desc=pd.DataFrame(descriptions); ev=pd.DataFrame(evidence)
    gap=pd.DataFrame(gaps);term=pd.DataFrame(terminal)
    ind.to_csv(OUT/'classification_by_admission.csv',index=False)
    desc.to_csv(OUT/'selected_action_descriptions.csv',index=False)
    ev.to_csv(OUT/'candidate_source_attribution.csv',index=False)
    gap.to_csv(OUT/'bridged_gaps.csv',index=False)
    term.to_csv(OUT/'unresolved_terminal_gaps.csv',index=False)
    transitions=ind.groupby(['scenario','reference_class','classification']).size().rename('n').reset_index()
    transitions.to_csv(OUT/'classification_transitions.csv',index=False)
    summary=[]
    for scenario in SCENARIOS:
        g=ind[ind.scenario==scenario]; old=g[g.reference_single]; cc=g.classification.value_counts()
        summary.append(dict(scenario=scenario,n=len(g),ne_first=int(cc.get(NE,0)),va_first=int(cc.get(VA,0)),
            simultaneous=int(cc.get('simultaneous',0)),none=int(cc.get('none',0)),
            no_accepted_action=int(cc.get('no_accepted_action',0)),paused_unclassified=int(cc.get('paused_unclassified',0)),
            class_agreement_n=int(g.same_class.sum()),class_agreement_pct=float(g.same_class.mean()*100),
            same_event_n=int(g.same_event.sum()),class_agreement_original284_n=int(old.same_class.sum()),
            same_event_original284_n=int(old.same_event.sum()),original_single_lost_n=int((~old.selected_single).sum()),
            new_single_from_other_n=int((~g.reference_single & g.selected_single).sum()),
            direct_order_switch_n=int(g.direct_order_switch.sum()),
            time_changed_among_finite_n=int(g.delay_from_reference_minutes.dropna().ne(0).sum()),
            same_class_time_changed_n=int((g.same_class & g.delay_from_reference_minutes.notna() & g.delay_from_reference_minutes.ne(0)).sum()),
            selected_with_virtual_other_drug_n=int((g.selected_single & ~g.actual_both_positive_before).sum()),
            complete_windows_n=int(g.original_window_complete.sum())))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'sensitivity_summary.csv',index=False)
    measures=[]
    for (scenario,cls),g in desc.groupby(['scenario','classification'],sort=False):
        q=dict(scenario=scenario,classification=cls,n=len(g),actual_both_positive_n=int(g.actual_both_positive_before.sum()))
        for col in ['ne_action','va_action','delay_h','ne_mean2h','va_mean2h']:
            q.update({col+'_median':float(g[col].median()),col+'_q25':float(g[col].quantile(.25)),col+'_q75':float(g[col].quantile(.75))})
        q['ne_down_va_flat_n']=int((g.ne_net2h.eq('net_down') & g.va_net2h.eq('approximately_flat')).sum())
        q['ne_flat_va_down_n']=int((g.ne_net2h.eq('approximately_flat') & g.va_net2h.eq('net_down')).sum())
        q['both_down_n']=int((g.ne_net2h.eq('net_down') & g.va_net2h.eq('net_down')).sum())
        q['incomplete_n']=int((g.ne_net2h.eq('incomplete') | g.va_net2h.eq('incomplete')).sum())
        measures.append(q)
    pd.DataFrame(measures).to_csv(OUT/'dose_pattern_summary.csv',index=False)
    desc.groupby(['scenario','classification','ne_net2h','va_net2h']).size().rename('n').reset_index().to_csv(OUT/'net_pattern_counts.csv',index=False)
    unique=ev[ev.scenario.eq('reference_selected')]
    unique.groupby(['classification','drug','status','paused']).size().rename('n').reset_index().to_csv(OUT/'reference_ending_status_counts.csv',index=False)
    prior=[]
    for name in ['core_manifest.json','weight_time_audit/manifest.json','design_feasibility/manifest.json',
                 'overlap_v1/manifest.json','nonlinear_dose_v1/manifest.json','descriptive_practice_v1/manifest.json',
                 'outcome28_audit_v1/manifest.json']:
        for entry in json.loads((SRC/name).read_text(encoding='utf-8')):
            p=Path(entry['path']);p=p if p.is_absolute() else BASE/p
            prior.append(dict(manifest=name,path=str(p),matches=sha(p)==entry['sha256']))
    for entry in json.loads((BASE/'docs/manuscript_practice_v3/manifest.json').read_text()):
        p=BASE/entry['path'];prior.append(dict(manifest='manuscript_practice_v3',path=str(p),matches=sha(p)==entry['sha256']))
    pd.DataFrame(prior).to_csv(OUT/'prior_manifest_verification.csv',index=False)
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','scripts','-p','test_cessation_sensitivity.py','-v'],cwd=BASE,capture_output=True,text=True)
    (OUT/'unit_test_output.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8')
    refdesc=desc[desc.scenario.eq('reference')].merge(pd.read_csv(SRC/'descriptive_practice_v1/decision_practice_individual.csv'),on='stay_id',suffixes=('','_prior'),validate='one_to_one')
    checks=dict(exact_reference470=all(refmatched),raw_timeline470_reproduced=all(reproduced),original_doses_unchanged=all(doseunchanged),
        fixed_population=all(len(g)==470 and g.stay_id.nunique()==470 and g.hadm_id.nunique()==470 and set(g.stay_id)==set(pool.stay_id) for _,g in ind.groupby('scenario')),
        six_locked_scenarios=ind.scenario.nunique()==6,
        transition_totals=transitions.groupby('scenario').n.sum().eq(470).all(),
        all_candidate_sources_positive=ev.normalized_rate.gt(0).all(),
        reference_doses_reproduced=np.allclose(refdesc[['ne_action','va_action']],refdesc[['ne_action_prior','va_action_prior']]),
        reference_patterns_reproduced=refdesc.ne_net2h.eq(refdesc.ne_net2h_prior).all() and refdesc.va_net2h.eq(refdesc.va_net2h_prior).all(),
        bridged_gaps_within_threshold=gap.gap_minutes.le(gap.threshold_minutes+1e-8).all() if len(gap) else True,
        no_earlier_reclassified_action=ind.delay_from_reference_minutes.dropna().ge(0).all(),
        original_manifests_unchanged=all(v['matches'] for v in prior),
        input_hashes_unchanged=all(sha(BASE/v['path'])==v['sha256'] for v in json.loads((OUT/'input_manifest.json').read_text())),
        ten_boundary_tests_pass=tests.returncode==0 and 'Ran 10 tests' in tests.stderr)
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(status='CESSATION_SENSITIVITY_CHECKS_PASS' if all(checks.values()) else 'REQUIRES_CORRECTION',checks=checks,
        source_status_tie_disagreement_rows=int(ev.status_tie_disagreement.sum()),
        baseline_paused_single_n=int(ind[(ind.scenario=='reference') & ind.reference_single].reference_paused.sum()),
        baseline_paused_all_action_n=int(ind[ind.scenario=='reference'].reference_paused.sum()),
        prior_manifest_entries_checked=len(prior),clinical_intent_verified=False,new_outcome_effects_computed=False,submission_ready=False)
    write_json(OUT/'validation.json',result)
    print(summary.to_string(index=False));print(json.dumps(result,indent=2));assert all(checks.values()),checks

if __name__=='__main__': main()
