"""Build strict postoperative stable cohort and exact-dose timelines from source rows."""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
from rebuild_v3_core import *

BASE = Path(__file__).resolve().parents[1]
OUT = BASE/'rebuild_v3'
OUT.mkdir(exist_ok=True)


def main():
    st = pd.read_csv(BASE/'data_derived/cohort_stays.csv')
    for c in ['intime','outtime','deathtime']:
        st[c] = pd.to_datetime(st[c], errors='coerce')
    px = pd.read_csv(BASE.parent/'hosp/procedures_icd.csv', dtype={'icd_code':str,'icd_version':str})
    dd = pd.read_csv(BASE.parent/'hosp/d_icd_procedures.csv', dtype=str)
    px = px.merge(dd, on=['icd_code','icd_version'], how='left')
    c = px.icd_code.str.replace('.','',regex=False)
    # Explicit conservative open-surgery phenotype; no service-only inference.
    nine = (px.icd_version=='9') & (c.str.startswith(('351','352','353','356','357','358','3591','3592','3593','3594','3595','361')) | c.isin(['3551','3553','3554','3631','3751','3752','3961']))
    ten = (px.icd_version=='10') & ((c.str.startswith(('0210','0211','0212','0213','02R','02Q','02U','02B','02T','02Y')) & (c.str[4]=='0')) | c.isin(['5A1221Z','5A1211Z']))
    px = px[nine|ten].copy()
    px['chartdate'] = pd.to_datetime(px.chartdate)
    px[['icd_code','icd_version','long_title']].drop_duplicates().to_csv(OUT/'surgery_codebook.csv',index=False)
    first = px.groupby('hadm_id').chartdate.min()
    st['surgery_date'] = st.hadm_id.map(first)
    st = st[(st.age>=18) & st.surgery_date.notna() & st.intime.notna() & st.outtime.notna()].copy()
    st['surgery_available'] = st.surgery_date + pd.Timedelta(days=1)
    pr = pd.read_pickle(BASE/'data_derived/pressor_rows.pkl')
    pr['starttime'] = pd.to_datetime(pr.starttime)
    pr['endtime'] = pd.to_datetime(pr.endtime)
    candidate_sids = set(st.stay_id)
    pr = pr[pr.stay_id.isin(candidate_sids) & pr.drug.isin(DRUGS)]
    groups = {sid:g for sid,g in pr.groupby('stay_id',sort=False)}
    pxgroups = {hid:g for hid,g in px.groupby('hadm_id',sort=False)}
    timelines, rows, exclusion = {}, [], []
    for r in st.itertuples():
        q = groups.get(r.stay_id)
        if q is None or not {NE,VA}.issubset(set(q.drug)):
            continue
        tl = build_timeline(q, r.intime, r.outtime)
        t0 = first_stable(tl, seconds(r.surgery_available))
        obs = min(seconds(r.outtime), seconds(r.deathtime) if pd.notna(r.deathtime) else np.inf)
        if t0 is None or t0 >= obs:
            exclusion.append(dict(stay_id=r.stay_id, reason='no_postoperative_stable_state_before_observation_end'))
            continue
        action, firstdrug = first_action(tl,t0,min(t0+43200,obs))
        if action >= obs:
            action,firstdrug = np.inf,'none'
        qp = q[(q.starttime < timestamp(t0)) & (q.patientweight.between(20,300))].sort_values('starttime')
        pp = pxgroups[r.hadm_id]
        pp = pp[pp.chartdate < timestamp(t0).normalize()]
        codes = pp.icd_code.str.replace('.','',regex=False)
        rows.append(dict(stay_id=r.stay_id,hadm_id=r.hadm_id,subject_id=r.subject_id,
            stable_t0=timestamp(t0),action_time=timestamp(action) if np.isfinite(action) else pd.NaT,
            first_drug=firstdrug,age=r.age,male=int(r.gender=='M'),
            weight=float(qp.patientweight.iloc[-1]) if len(qp) else np.nan,
            intime=r.intime,outtime=r.outtime,deathtime=r.deathtime,
            surgery_date=r.surgery_date,surgery_available=r.surgery_available,
            cabg=int(codes.str.startswith(('361','0210','0211','0212','0213')).any()),
            valve=int(codes.str.startswith(('351','352','353','02R','02Q','02U')).any()),
            cpb=int(codes.isin(['3961','5A1221Z','5A1211Z']).any())))
        timelines[r.stay_id] = tl
    pool = pd.DataFrame(rows).sort_values(['stable_t0','stay_id']).drop_duplicates('hadm_id',keep='first').reset_index(drop=True)
    pool.to_csv(OUT/'stable_pool.csv',index=False)
    timelines = {int(s):timelines[s] for s in pool.stay_id}
    with (OUT/'timelines.pkl').open('wb') as f:
        pickle.dump(timelines,f)
    pr[pr.stay_id.isin(pool.stay_id)].to_pickle(OUT/'source_pressors.pkl')
    pd.DataFrame(exclusion).to_csv(OUT/'cohort_exclusions.csv',index=False)
    dec = pool[pool.first_drug.isin([NE,VA])].copy()
    dec.to_csv(OUT/'decision_pool.csv',index=False)
    assert (pd.to_datetime(dec.action_time)>=pd.to_datetime(dec.stable_t0)).all()
    assert set(dec.stay_id).issubset(pool.stay_id)
    orig = pd.read_csv(BASE/'results/tables/episodes_primary.csv')
    report = dict(strict_surgery_adult_stays=len(st),stable_stays_before_hadm_dedup=len(rows),
        stable_pool_n=len(pool),stable_subjects=int(pool.subject_id.nunique()),
        decision_pool_n=len(dec),decision_groups=dec.first_drug.value_counts().to_dict(),
        simultaneous_n=int((pool.first_drug=='simultaneous').sum()),
        no_action_within12h_n=int((pool.first_drug=='none').sum()),
        legacy393_overlap_stays=len(set(orig.stay_id)&set(dec.stay_id)),
        status='COHORT_BUILT_PENDING_INFERENCE')
    (OUT/'cohort_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Preserve protocol and input content hashes before inference.
    paths = [BASE/'docs/rebuild_protocol_v3_20260914.md',BASE/'data_derived/pressor_rows.pkl',BASE/'data_derived/cohort_stays.csv',BASE.parent/'hosp/procedures_icd.csv',BASE.parent/'hosp/d_icd_procedures.csv']
    manifest=[]
    for p in paths:
        with p.open('rb') as f:
            sha=hashlib.file_digest(f,'sha256').hexdigest()
        manifest.append(dict(path=str(p),sha256=sha,bytes=p.stat().st_size))
    (OUT/'cohort_input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
