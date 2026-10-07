"""Audit recorded 28-day endpoints from all linked raw stays and admissions.
No comparative fitting; no support-free-day imputation outside ICU.
"""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
import duckdb
from outcome28_core import death_status, union_seconds
from rebuild_v3_core import build_timeline, window, seconds

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'rebuild_v3/outcome28_audit_v1'
ITEMS={221906:'norepinephrine',222315:'vasopressin',221289:'epinephrine',229617:'epinephrine',
       221662:'dopamine',221749:'phenylephrine',229630:'phenylephrine',229632:'phenylephrine',
       229709:'angiotensin_ii',229764:'angiotensin_ii'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    sources=[BASE.parent/p for p in ['hosp/patients.csv','hosp/admissions.csv','icu/icustays.csv','icu/inputevents.csv']]
    source_stats={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in sources}
    pool=pd.read_csv(BASE/'rebuild_v3/stable_pool.csv')
    decision=pd.read_csv(BASE/'rebuild_v3/decision_analysis.csv')
    anchors=pd.concat([pool[['subject_id','hadm_id','stay_id','stable_t0']].rename(columns={'stable_t0':'t0'}).assign(anchor='stable',A=-1),
                       decision[['subject_id','hadm_id','stay_id','t0','A']].assign(anchor='action')],ignore_index=True)
    anchors['t0']=pd.to_datetime(anchors.t0); anchors['day28']=anchors.t0+pd.Timedelta(days=28)
    pat=pd.read_csv(sources[0]); adm=pd.read_csv(sources[1]); icu=pd.read_csv(sources[2])
    pat=pat[pat.subject_id.isin(pool.subject_id)].copy()
    adm=adm[adm.subject_id.isin(pool.subject_id)].copy()
    icu=icu[icu.subject_id.isin(pool.subject_id)].copy()
    for frame,cols in [(pat,['dod']),(adm,['admittime','dischtime','deathtime']),(icu,['intime','outtime'])]:
        for col in cols: frame[col]=pd.to_datetime(frame[col],errors='coerce')
    pat.to_csv(OUT/'source_patients.csv',index=False); adm.to_csv(OUT/'source_admissions.csv',index=False)
    icu.to_csv(OUT/'source_all_patient_icustays.csv',index=False)
    pp=pat.set_index('subject_id'); ag={s:g for s,g in adm.groupby('subject_id')}; ig={s:g for s,g in icu.groupby('subject_id')}
    relevant=set()
    for r in anchors.itertuples():
        q=ig[r.subject_id]; relevant.update(q.loc[(q.intime<r.day28)&(q.outtime>r.t0),'stay_id'])
    con=duckdb.connect(); con.execute('SET threads=4'); con.execute("SET memory_limit='2GB'")
    con.register('selected_stays',pd.DataFrame({'stay_id':sorted(relevant)}))
    print('Extracting raw infusion rows for',len(relevant),'window-relevant ICU stays',flush=True)
    sql="SELECT x.* FROM read_csv(?,header=true,all_varchar=true) x JOIN selected_stays s ON CAST(x.stay_id AS BIGINT)=s.stay_id WHERE CAST(x.itemid AS BIGINT) IN ("+','.join(map(str,ITEMS))+')'
    pr=con.execute(sql,[str(sources[3])]).fetchdf(); con.close()
    for col in ['subject_id','hadm_id','stay_id','itemid']: pr[col]=pd.to_numeric(pr[col])
    for col in ['rate','patientweight']: pr[col]=pd.to_numeric(pr[col],errors='coerce')
    for col in ['starttime','endtime','storetime']: pr[col]=pd.to_datetime(pr[col],errors='coerce')
    pr['drug']=pr.itemid.map(ITEMS)
    pr=pr.sort_values(['stay_id','starttime','itemid','endtime','rate'],kind='stable').reset_index(drop=True)
    pr.to_csv(OUT/'source_window_pressors.csv',index=False)
    pg={s:g for s,g in pr.groupby('stay_id')}; tls={}
    for r in icu[icu.stay_id.isin(relevant)].itertuples():
        tls[r.stay_id]=build_timeline(pg.get(r.stay_id,pr.iloc[:0]),r.intime,r.outtime)
    records=[]; contacts=[]
    for r in anchors.itertuples():
        aa,ii=ag[r.subject_id],ig[r.subject_id]
        p=pp.loc[r.subject_id]; death_times=aa.deathtime.dropna().unique()
        exact=pd.Timestamp(min(death_times)) if len(death_times) else pd.NaT
        conflict=len(death_times)>1 or (pd.notna(exact) and pd.notna(p.dod) and exact.normalize()!=p.dod.normalize())
        # New admission after the full date-only death interval also contradicts death chronology.
        late_admission=bool((aa.admittime>(exact if pd.notna(exact) else p.dod+pd.Timedelta(days=1))).any()) if pd.notna(exact) or pd.notna(p.dod) else False
        conflict=conflict or late_admission
        coverage=aa.dischtime.max()+pd.Timedelta(days=365)
        status=death_status(r.t0,r.day28,exact,p.dod,coverage,conflict)
        a,z=seconds(r.t0),seconds(r.day28)
        index=ii[ii.stay_id==r.stay_id].iloc[0]
        q=ii[(ii.intime<r.day28)&(ii.outtime>r.t0)]
        later=q[(q.stay_id!=r.stay_id)&(q.intime>=index.outtime)]
        icu_s=union_seconds([(seconds(x.intime),seconds(x.outtime)) for x in q.itertuples()],a,z)
        hosp_s=union_seconds([(seconds(x.admittime),seconds(x.dischtime)) for x in aa.itertuples() if pd.notna(x.dischtime)],a,z)
        ip,iu=[],[]
        for x in q.itertuples():
            lo,hi,v=window(tls[x.stay_id],a,z)
            positive=(np.nan_to_num(v,nan=0)>0).any(axis=1)
            unknown=(~np.isfinite(v)).any(axis=1)
            ip.extend(zip(lo[positive],hi[positive])); iu.extend(zip(lo[unknown],hi[unknown]))
        pos_s=union_seconds(ip,a,z); unknown_s=union_seconds(iu,a,z)
        index_s=max(0,min(z,seconds(index.outtime))-a)
        admindex=aa[aa.hadm_id==r.hadm_id].iloc[0]
        after_death_s=union_seconds([(seconds(x.intime),seconds(x.outtime)) for x in q.itertuples()],max(a,seconds(exact)),z) if pd.notna(exact) and a<seconds(exact)<z else 0.
        records.append(dict(anchor=r.anchor,A=r.A,subject_id=r.subject_id,hadm_id=r.hadm_id,stay_id=r.stay_id,t0=r.t0,day28=r.day28,
            death_status=status,death_exact=exact,dod=p.dod,last_hospital_discharge=aa.dischtime.max(),linkage_coverage_end_lower_bound=coverage,
            linkage_window_covers28=int(r.day28<=coverage),death_record_conflict=int(conflict),new_admission_after_death=int(late_admission),
            death28=int(status.startswith('DEATH_WITHIN28')) if status.startswith(('DEATH_WITHIN28','NO_DEATH')) else np.nan,
            hospital_death_missing_timestamp=int(((aa.hospital_expire_flag==1)&aa.deathtime.isna()).any()),
            index_hospital_discharge_location=admindex.discharge_location,index_hospital_discharge=admindex.dischtime,
            index_icu_end=index.outtime,index_icu_end_before28=int(index.outtime<r.day28),
            subsequent_icu_contact28=int(len(later)>0),subsequent_icu_contacts_n=len(later),
            subsequent_same_hadm28=int((later.hadm_id==r.hadm_id).any()),subsequent_different_hadm28=int((later.hadm_id!=r.hadm_id).any()),
            overlapping_other_stay_n=int(((q.stay_id!=r.stay_id)&(q.intime<index.outtime)).sum()),
            recorded_icu_days28=icu_s/86400,recorded_index_icu_days28=index_s/86400,additional_icu_days28=(icu_s-index_s)/86400,
            outside_recorded_icu_days28=28-icu_s/86400,recorded_hospital_days28=hosp_s/86400,outside_recorded_hospital_days28=28-hosp_s/86400,
            recorded_positive_pressor_days28=pos_s/86400,unknown_active_rate_days28=unknown_s/86400,
            icu_admin_days_after_exact_death=after_death_s/86400))
        for x in later.itertuples():
            contacts.append(dict(anchor=r.anchor,anchor_stay_id=r.stay_id,subject_id=r.subject_id,contact_stay_id=x.stay_id,
                contact_hadm_id=x.hadm_id,same_hadm=int(x.hadm_id==r.hadm_id),intime=x.intime,outtime=x.outtime,gap_from_index_icu_hours=(x.intime-index.outtime).total_seconds()/3600))
    result=pd.DataFrame(records); result.to_csv(OUT/'observability_by_admission.csv',index=False)
    pd.DataFrame(contacts).to_csv(OUT/'subsequent_icu_contacts.csv',index=False)
    summaries=[]
    groups=[('stable_all',result[result.anchor=='stable']),('action_all',result[result.anchor=='action']),
            ('action_ne_first',result[(result.anchor=='action')&(result.A==0)]),('action_va_first',result[(result.anchor=='action')&(result.A==1)])]
    for name,q in groups:
        row=dict(group=name,n=len(q),subjects=q.subject_id.nunique(),death28=int(q.death28.sum()),death28_unknown=int(q.death28.isna().sum()),
                 linkage_covers28=int(q.linkage_window_covers28.sum()),death_conflicts=int(q.death_record_conflict.sum()),
                 death28_exact=int((q.death_status=='DEATH_WITHIN28_EXACT').sum()),death28_date=int((q.death_status=='DEATH_WITHIN28_DATE').sum()),
                 subsequent_icu_contact28=int(q.subsequent_icu_contact28.sum()),subsequent_same_hadm28=int(q.subsequent_same_hadm28.sum()),
                 subsequent_different_hadm28=int(q.subsequent_different_hadm28.sum()),subsequent_icu_contacts_n=int(q.subsequent_icu_contacts_n.sum()),
                 index_icu_end_before28=int(q.index_icu_end_before28.sum()),any_time_outside_recorded_icu=int((q.outside_recorded_icu_days28>1e-8).sum()),
                 any_unknown_active_rate=int((q.unknown_active_rate_days28>1e-8).sum()),icu_admin_after_exact_death=int((q.icu_admin_days_after_exact_death>0).sum()))
        for c in ['recorded_icu_days28','recorded_index_icu_days28','additional_icu_days28','outside_recorded_icu_days28','recorded_hospital_days28','outside_recorded_hospital_days28','recorded_positive_pressor_days28','unknown_active_rate_days28']:
            for stat,v in zip(['q25','median','q75'],q[c].quantile([.25,.5,.75])): row[c+'_'+stat]=float(v)
        summaries.append(row)
    summary=pd.DataFrame(summaries); summary.to_csv(OUT/'summary.csv',index=False)
    result.groupby(['anchor','A','death_status']).size().rename('n').reset_index().to_csv(OUT/'death_status_counts.csv',index=False)
    result.groupby(['anchor','A','index_hospital_discharge_location']).size().rename('n').reset_index().to_csv(OUT/'discharge_location_counts.csv',index=False)
    # Compare exact rebuilt index time lines against the immutable earlier extraction.
    original=pickle.loads((BASE/'rebuild_v3/timelines.pkl').read_bytes())
    equal=all(np.array_equal(tls[s]['bounds'],original[s]['bounds']) and np.allclose(tls[s]['rates'],original[s]['rates'],equal_nan=True) for s in pool.stay_id)
    checks=dict(anchor_counts=len(result)==754 and len(pool)==470 and len(decision)==284,
        unique_patient_master=pat.subject_id.is_unique,all_anchor_ids_linked=anchors.stay_id.isin(icu.stay_id).all(),
        index_timeline_raw_reproduction=equal,source_intervals_complete=icu[['intime','outtime']].notna().all().all(),
        source_icu_intervals_positive=(icu.outtime>icu.intime).all(),death_records_consistent=not result.death_record_conflict.any(),
        hospital_death_timestamps_present=not result.hospital_death_missing_timestamp.any(),
        death_status_no_unresolved=result.death28.notna().all(),linkage_window_covers_all=result.linkage_window_covers28.eq(1).all(),
        no_overlapping_index_stays=not result.overlapping_other_stay_n.any(),icu_union_bounded=result.recorded_icu_days28.between(0,28+1e-9).all(),
        hospital_union_bounded=result.recorded_hospital_days28.between(0,28+1e-9).all(),index_within_union=(result.additional_icu_days28>=-1e-9).all(),
        pressor_within_icu=(result.recorded_positive_pressor_days28<=result.recorded_icu_days28+1e-9).all(),
        unknown_within_icu=(result.unknown_active_rate_days28<=result.recorded_icu_days28+1e-9).all(),
        sources_unchanged=all(source_stats[str(p)]==[p.stat().st_size,p.stat().st_mtime_ns] for p in sources))
    # Verify every file in core, descriptive and nonlinear previous manifests remains unchanged.
    old=[]
    for rel in ['core_manifest.json','descriptive_practice_v1/manifest.json','nonlinear_dose_v1/manifest.json']:
        for e in json.loads((BASE/'rebuild_v3'/rel).read_text(encoding='utf-8')):
            path=BASE/e['path']; old.append(path.exists() and sha(path)==e['sha256'])
    checks['previous_manifests_preserved']=all(old)
    validation={'status':'OUTCOME28_OBSERVABILITY_CHECKS_PASS' if all(checks.values()) else 'OUTCOME28_REQUIRES_REVIEW',
                'checks':{k:bool(v) for k,v in checks.items()},'bounds':'Checks concern recorded data, not completeness of death linkage or external care.'}
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    manifest=[]
    for p in sources:
        manifest.append(dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=sha(p)))
    for p in [BASE/'rebuild_v3/stable_pool.csv',BASE/'rebuild_v3/decision_analysis.csv',BASE/'docs/outcome28_observability_plan_v1.md']:
        manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(summary[['group','n','death28','death28_unknown','subsequent_icu_contact28','index_icu_end_before28','any_unknown_active_rate']].to_string(index=False))
    print(json.dumps(validation,indent=2),flush=True)

if __name__=='__main__': main()
