"""Stream raw tables for the NEW cohort; cache provenance includes cohort hash."""
from pathlib import Path
import hashlib
import json
import time
import pandas as pd
import duckdb

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'rebuild_v3'


def main():
    pool=pd.read_csv(OUT/'stable_pool.csv')
    sids,hids=set(pool.stay_id),set(pool.hadm_id)
    configs=[
        ('chart','icu/chartevents.csv','stay_id',sids,[220052,220181,225312,220045],['stay_id','itemid','charttime','storetime','valuenum']),
        ('lab','hosp/labevents.csv','hadm_id',hids,[50813,50912,51265,51300,51301,51222,51006],['hadm_id','itemid','charttime','storetime','valuenum']),
        ('urine','icu/outputevents.csv','stay_id',sids,[226559,226560,226561,226584,226563,226564,226565,226567,226557,226558,227488,227489],['stay_id','itemid','charttime','storetime','value']),
        ('procedure','icu/procedureevents.csv','stay_id',sids,None,['stay_id','itemid','starttime','endtime'])]
    cohort_hash=hashlib.sha256((OUT/'stable_pool.csv').read_bytes()).hexdigest()
    records=[]
    for name,rel,idcol,ids,itemids,cols in configs:
        target=OUT/f'raw_{name}.pkl'
        source=BASE.parent/rel
        sig=dict(cohort_sha256=cohort_hash,source=str(source),size=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns,columns=cols,itemids=itemids,engine='duckdb',covered_ids=sorted(int(x) for x in ids))
        meta=OUT/f'raw_{name}_meta.json'
        if target.exists() and meta.exists() and json.loads(meta.read_text())['signature']==sig:
            print(name,'cached',flush=True)
            records.append(json.loads(meta.read_text()))
            continue
        started=time.monotonic()
        con=duckdb.connect()
        con.execute("SET threads=4")
        con.execute("SET memory_limit='2GB'")
        con.register('eligible_ids',pd.DataFrame({'eligible_id':sorted(ids)}))
        selected=[]
        for c in cols:
            kind='BIGINT' if c in ['stay_id','hadm_id','itemid'] else ('DOUBLE' if c in ['value','valuenum'] else 'TIMESTAMP')
            selected.append(f'CAST(x.{c} AS {kind}) AS {c}')
        filt=''
        if itemids is not None:filt=' WHERE CAST(x.itemid AS BIGINT) IN ('+','.join(map(str,itemids))+')'
        sql='SELECT '+','.join(selected)+' FROM read_csv(?,header=true,all_varchar=true,parallel=true) x JOIN eligible_ids e ON CAST(x.'+idcol+' AS BIGINT)=e.eligible_id'+filt
        print(name,'streaming source with DuckDB',flush=True)
        data=con.execute(sql,[str(source)]).fetchdf()
        con.close()
        for c in ['charttime','storetime','starttime','endtime']:
            if c in data:data[c]=pd.to_datetime(data[c],errors='coerce')
        data.to_pickle(target)
        record=dict(signature=sig,scanned='full source scan; row count not collected',kept=len(data),elapsed_s=round(time.monotonic()-started,2))
        meta.write_text(json.dumps(record,indent=2),encoding='utf-8');records.append(record)
        print(name,record['kept'],'DONE',flush=True)
    (OUT/'extraction_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()
