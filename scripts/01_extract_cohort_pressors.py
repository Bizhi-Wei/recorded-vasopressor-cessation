# -*- coding: utf-8 -*-
"""
01. 提取心脏术后队列 + 升压药 infusion 记录（MIMIC-IV v3.1）

输出（data_derived/）:
  - cohort_stays.csv        所有 ICU stay + 心脏手术/CSURG/CVICU 旗标
  - pressor_rows.pkl        升压药 infusion 原始行（已按 itemid 过滤）

心脏手术口径（primary, 审计见 logs/01_*.log）:
  ICD-9-CM 手术码（去小数点后前缀）:
    '35'     心脏瓣膜/间隔手术
    '361'    冠脉搭桥 (CABG)，ICD-9 36.1x
    '363'    心脏血管动脉瘤修补 (36.3x)
    '3751'   心脏移植, '3752' 植入心脏辅助装置
    '3961'   体外循环辅助开心手术 (CPB)
    注意: ICD-9 '36.0x' 是 PTCA/冠脉支架（经皮），必须排除
  ICD-10-PCS:
    '0210'-'0213' 冠脉搭桥
    '02R'    心脏瓣膜置换（排除 Percutaneous Approach，如 TAVR）
    '02Q'    心脏间隔修补, '02U' 瓣膜补片成形, '02B' 瓣膜切除, '02T' 腱索
    '02Y'    心脏移植
    '5A1221Z'/'5A1211Z'  体外循环 (CPB, 标签为 Performance of Cardiac Output)
  secondary 口径（仅用于对比）: hosp.services 的 CSURG；first_careunit = CVICU
"""
import os
import sys
import time
import pandas as pd

ROOT = r"D:\mimic-iv-3.1"
OUT = r"D:\mimic-iv-3.1\vasopressor_sequencing_tte\data_derived"
os.makedirs(OUT, exist_ok=True)

# 升压药 itemid（含正性肌力药之外的血管活性药）
PRESSOR_ITEMS = {
    221906: "norepinephrine",
    222315: "vasopressin",
    221289: "epinephrine",
    229617: "epinephrine",
    221662: "dopamine",
    221749: "phenylephrine",
    229630: "phenylephrine",
    229632: "phenylephrine",
    229709: "angiotensin_ii",
    229764: "angiotensin_ii",
}

ICD9_PREFIX = ("35", "361", "363", "3751", "3752", "3961")
ICD10_PREFIX = ("0210", "0211", "0212", "0213", "02R", "02Q", "02U", "02B",
                "02T", "02Y", "5A1221Z", "5A1211Z")


def log(msg, fh):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")


def main():
    logf = open(os.path.join(r"D:\mimic-iv-3.1\vasopressor_sequencing_tte\logs",
                             f"01_extract_{time.strftime('%Y%m%d_%H%M')}.log"),
                "w", encoding="utf-8")

    # ---------- 1. ICU stays / patients / admissions ----------
    icu = pd.read_csv(os.path.join(ROOT, "icu", "icustays.csv"), low_memory=False)
    pat = pd.read_csv(os.path.join(ROOT, "hosp", "patients.csv"), low_memory=False)
    adm = pd.read_csv(os.path.join(ROOT, "hosp", "admissions.csv"), low_memory=False,
                      usecols=["hadm_id", "admittime", "dischtime", "deathtime",
                               "hospital_expire_flag", "admission_type"])
    icu = icu.merge(pat[["subject_id", "gender", "anchor_age", "anchor_year", "dod"]],
                    on="subject_id", how="left")
    icu = icu.merge(adm, on="hadm_id", how="left")
    log(f"icustays={len(icu)} subjects={icu.subject_id.nunique()} hadm={icu.hadm_id.nunique()}", logf)

    # ---------- 2. 心脏手术 hadm ----------
    px = pd.read_csv(os.path.join(ROOT, "hosp", "procedures_icd.csv"), low_memory=False,
                     dtype={"icd_code": str, "icd_version": str})
    dd = pd.read_csv(os.path.join(ROOT, "hosp", "d_icd_procedures.csv"), dtype=str,
                     low_memory=False)
    dd["icd_version"] = dd["icd_version"].astype(str)
    px = px.merge(dd, on=["icd_code", "icd_version"], how="left")
    px["code_nd"] = px["icd_code"].str.replace(".", "", regex=False)
    px["title"] = px["long_title"].fillna("")

    nine = px[px.icd_version == "9"]
    ten = px[px.icd_version == "10"]
    hit9 = nine[nine.code_nd.str.startswith(ICD9_PREFIX)]
    hit10 = ten[
        ten.code_nd.str.startswith(ICD10_PREFIX)
        & ~ten.title.str.contains("Percutaneous Approach", case=False, na=False)
    ]
    cs_hadm = set(hit9.hadm_id) | set(hit10.hadm_id)
    log(f"cardiac surgery ICD: icd9_rows={len(hit9)} icd10_rows={len(hit10)} "
        f"hadm={len(cs_hadm)}", logf)
    log("ICD-9 命中码 top:", logf)
    for c, n in hit9.code_nd.str.slice(0, 4).value_counts().head(12).items():
        log(f"    {c}: {n}", logf)
    log("ICD-10 命中码 top:", logf)
    for c, n in hit10.code_nd.str.slice(0, 4).value_counts().head(12).items():
        log(f"    {c}: {n}", logf)

    svc = pd.read_csv(os.path.join(ROOT, "hosp", "services.csv"), low_memory=False)
    csur_hadm = set(svc.loc[svc.curr_service == "CSURG", "hadm_id"])

    icu["cardiac_surgery_icd"] = icu.hadm_id.isin(cs_hadm)
    icu["csurg_service"] = icu.hadm_id.isin(csur_hadm)
    icu["cvicu"] = icu.first_careunit.eq("Cardiac Vascular Intensive Care Unit (CVICU)")
    icu["cs_any"] = icu.cardiac_surgery_icd | icu.csurg_service | icu.cvicu

    log(f"stays: icd={icu.cardiac_surgery_icd.sum()} csurg={icu.csurg_service.sum()} "
        f"cvicu={icu.cvicu.sum()} any={icu.cs_any.sum()}", logf)
    log("交叉表 (icd vs csurg vs cvicu):\n"
        + str(icu.groupby(["cardiac_surgery_icd", "csurg_service", "cvicu"]).size()), logf)

    # anchor_age 为 89+ 的按 MIMIC 规则（>=89 记 91）; 实际年龄 = anchor_age + (year-intime - anchor_year)
    icu["age"] = icu.anchor_age + (pd.to_datetime(icu.intime).dt.year - icu.anchor_year)

    keep = ["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit",
            "intime", "outtime", "los", "gender", "age", "admittime", "dischtime",
            "deathtime", "hospital_expire_flag", "admission_type",
            "cardiac_surgery_icd", "csurg_service", "cvicu", "cs_any"]
    icu[keep].to_csv(os.path.join(OUT, "cohort_stays.csv"), index=False)
    log("saved cohort_stays.csv", logf)

    # ---------- 3. 升压药 infusion 行 ----------
    t0 = time.time()
    chunks = []
    reader = pd.read_csv(
        os.path.join(ROOT, "icu", "inputevents.csv"), low_memory=False, chunksize=2_000_000,
        usecols=["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid",
                 "rate", "rateuom", "statusdescription", "orderid", "linkorderid",
                 "patientweight"],
        dtype={"itemid": "int32"},
    )
    n_raw = 0
    for ch in reader:
        n_raw += len(ch)
        sub = ch[ch.itemid.isin(PRESSOR_ITEMS)]
        if len(sub):
            chunks.append(sub)
    pr = pd.concat(chunks, ignore_index=True)
    pr["drug"] = pr.itemid.map(PRESSOR_ITEMS)
    pr.to_pickle(os.path.join(OUT, "pressor_rows.pkl"))
    log(f"inputevents rows={n_raw} pressor rows={len(pr)} "
        f"({time.time()-t0:.0f}s) stays={pr.stay_id.nunique()}", logf)
    log("按药/单位:\n" + str(pr.groupby(["drug", "rateuom"]).size()), logf)
    log("按药/stay 数:\n" + str(pr.groupby("drug").stay_id.nunique()), logf)
    logf.close()


if __name__ == "__main__":
    main()
