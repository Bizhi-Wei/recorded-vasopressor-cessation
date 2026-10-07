"""Read original high-NE rows; compare dose with documented amount and duration."""
from pathlib import Path
import json
import hashlib
import duckdb
import pandas as pd

BASE=Path(__file__).resolve().parents[1];OUT=BASE/'rebuild_v3/source_rate_audit'
def main():
    OUT.mkdir(exist_ok=True)
    d=pd.read_csv(BASE/'rebuild_v3/decision_analysis.csv')
    row=d.loc[d['ne'].idxmax()];sid=str(int(row.stay_id))
    source=BASE.parent/'icu/inputevents.csv'
    con=duckdb.connect();con.execute('SET threads=4')
    raw=con.execute("SELECT * FROM read_csv_auto(?, all_varchar=true) WHERE stay_id=? AND itemid='221906'",[str(source),sid]).fetchdf()
    raw.to_csv(OUT/'raw_high_ne_stay.csv',index=False)
    for col in ['starttime','endtime','storetime']:raw[col]=pd.to_datetime(raw[col])
    for col in ['rate','amount','patientweight','originalrate','originalamount']:raw[col]=pd.to_numeric(raw[col],errors='coerce')
    raw['duration_min']=(raw.endtime-raw.starttime).dt.total_seconds()/60
    raw['rate_predicted_amount_mg']=raw.rate*raw.patientweight*raw.duration_min/1000
    raw['amount_matches_rate_assuming_mg']=((raw.amount-raw.rate_predicted_amount_mg).abs()/raw.amount.abs()).where(raw.amount.ne(0))
    action=pd.Timestamp(row.t0)
    near=raw[raw.starttime.lt(action+pd.Timedelta(hours=2))&raw.endtime.gt(action-pd.Timedelta(hours=2))].copy()
    near.to_csv(OUT/'adjacent_record_consistency.csv',index=False)
    source_meta=dict(path=str(source),bytes=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns,
        selected_stay_id=int(row.stay_id),selection='largest decision-time NE dose; diagnostic only',
        raw_filtered_sha256=hashlib.sha256((OUT/'raw_high_ne_stay.csv').read_bytes()).hexdigest(),
        changed_cohort=False,changed_rate=False)
    (OUT/'source_manifest.json').write_text(json.dumps(source_meta,indent=2),encoding='utf-8')
    print(near[['starttime','endtime','storetime','rate','rateuom','amount','amountuom','originalrate','originalamount','duration_min','rate_predicted_amount_mg','statusdescription']].to_string(index=False),flush=True)
if __name__=='__main__':main()
