"""Shared event-time functions; no model fitting or filesystem mutation at import."""
import numpy as np
import pandas as pd

NE, VA = 'norepinephrine', 'vasopressin'
DRUGS = [NE, VA, 'epinephrine', 'phenylephrine', 'dopamine', 'angiotensin_ii']
COEF = np.array([1., 2.5/60, 1., .06, .01, 2.5])
CAP = dict(zip(DRUGS, [5., 12., 5., 20., 50., .2]))
EPS = 1e-7


def seconds(x):
    return pd.Timestamp(x).value / 1e9


def timestamp(x):
    return pd.Timestamp(x, unit='s')


def normalize_rate(drug, rate, unit, weight):
    if pd.isna(rate) or not np.isfinite(rate):
        return np.nan
    if drug == VA:
        fac = {'units/hour': 1., 'units/min': 60.}.get(unit)
    else:
        fac = {'mcg/kg/min': 1., 'mg/kg/min': 1000., 'ng/kg/min': .001}.get(unit)
        if fac is None and np.isfinite(weight) and weight > 0:
            fac = {'mcg/min': 1./weight, 'mg/min': 1000./weight}.get(unit)
    val = rate*fac if fac is not None else np.nan
    return val if np.isfinite(val) and 0 <= val <= CAP[drug] else np.nan


def build_timeline(rows, intime, outtime):
    """Unknown active rows remain NaN; absent infusion records imply zero."""
    start, end = seconds(intime), seconds(outtime)
    work = rows.copy()
    work['s'] = pd.to_datetime(work.starttime).to_numpy(dtype='datetime64[ns]').astype('int64')/1e9
    work['e'] = pd.to_datetime(work.endtime).to_numpy(dtype='datetime64[ns]').astype('int64')/1e9
    work = work[work.statusdescription.isin(['ChangeDose/Rate', 'Changed', 'FinishedRunning', 'Stopped', 'Paused']) & (work.e > work.s)]
    work['original_s'] = work.s
    work['s'] = work.s.clip(lower=start)
    work['e'] = work.e.clip(upper=end)
    work = work[work.e > work.s]
    b = np.unique(np.r_[start, end, work.s, work.e])
    v = np.zeros((len(b)-1, len(DRUGS)))
    for j, drug in enumerate(DRUGS):
        q = work[work.drug == drug].sort_values(['original_s', 'rate'], kind='stable')
        for r in q.itertuples():
            i, k = np.searchsorted(b, [r.s, r.e])
            v[i:k, j] = normalize_rate(drug, r.rate, r.rateuom, r.patientweight)
    return {'bounds': b, 'rates': v}


def at(tl, t, left=False):
    b, v = tl['bounds'], tl['rates']
    i = np.searchsorted(b, t, side='left' if left else 'right')-1
    return v[i].copy() if 0 <= i < len(v) else np.zeros(len(DRUGS))


def window(tl, a, z):
    b, v = tl['bounds'], tl['rates']
    lo, hi = np.maximum(b[:-1], a), np.minimum(b[1:], z)
    use = hi > lo
    return lo[use], hi[use], v[use]


def first_stable(tl, surgery_available):
    b, v = tl['bounds'], tl['rates']
    ok = np.isfinite(v[:,:2]).all(axis=1) & (v[:,:2] > 0).all(axis=1) & (v[:,0] <= .1001) & (v[:,1] <= 3.003)
    run = None
    for i in range(len(v)):
        if not ok[i]:
            run = None
            continue
        if run is None:
            run = b[i]
        t = max(run + 7200, surgery_available)
        if t <= b[i+1] and t < b[-1]:
            return t
    return None


def first_action(tl, t0, end):
    b = tl['bounds']
    for t in b[(b >= t0) & (b < min(end, b[-1])+EPS)]:
        if t >= b[-1]:
            continue
        before, after = at(tl, t, True), at(tl, t)
        # Both agents must be known and positive just before the first cessation.
        if not np.isfinite(before[:2]).all() or not (before[:2] > 0).all():
            continue
        if not np.isfinite(after[:2]).all():
            continue
        stopped = np.flatnonzero((before[:2] > 0) & (after[:2] == 0))
        if len(stopped):
            return float(t), ('simultaneous' if len(stopped) == 2 else DRUGS[stopped[0]])
    return np.inf, 'none'


def ned(tl, t, left=False):
    return float(at(tl, t, left) @ COEF)


def summarize_ned(tl, a, z):
    lo, hi, v = window(tl, a, z)
    y = v @ COEF
    if not len(y) or not np.isfinite(y).all():
        return np.nan, np.nan
    return float(y.max()), float(np.average(y, weights=hi-lo))


def first_unknown(tl, t0, end):
    lo, _, v = window(tl, t0, end)
    bad = ~np.isfinite(v).all(axis=1)
    return float(lo[bad][0]) if bad.any() else np.inf


def rescue_time(tl, action, first_drug, end):
    """Decision-anchored restart/escalation/new-drug composite; t0 exclusive."""
    if first_drug not in (NE, VA):
        return np.inf, 'none'
    j = DRUGS.index(first_drug)
    k = 1-j
    ref = at(tl, action, True)[k]
    threshold = max(1.5*ref, ref + (.02 if k == 0 else .6))
    _, _, pre = window(tl, action-21600, action)
    prior = (pre > 0).any(axis=0) if len(pre) else np.zeros(6, bool)
    v0 = at(tl, action)
    if v0[k] >= threshold-1e-12 or ((v0[2:] > 0) & ~prior[2:]).any():
        return float(action), 'concurrent_escalation_or_new_drug'
    for t in tl['bounds'][(tl['bounds'] > action) & (tl['bounds'] < end)]:
        v = at(tl, t)
        if v[j] > 0:
            return float(t), 'restart'
        if v[k] >= threshold-1e-12:
            return float(t), 'escalation'
        if ((v[2:] > 0) & ~prior[2:]).any():
            return float(t), 'new_drug'
    return np.inf, 'none'


def deterioration_time(tl, t0, action, first_drug, end):
    ref = ned(tl, t0, True)
    threshold = max(1.5*ref, ref+.02)
    _, _, pre = window(tl, t0-21600, t0)
    prior = (pre > 0).any(axis=0) if len(pre) else np.zeros(6, bool)
    times = np.unique(np.r_[t0, tl['bounds'][(tl['bounds'] >= t0) & (tl['bounds'] < end)]])
    for t in times:
        v = at(tl, t)
        if v @ COEF >= threshold-1e-12:
            return float(t), 'ned_escalation'
        if ((v[2:] > 0) & ~prior[2:]).any():
            return float(t), 'new_drug'
        if first_drug in (NE, VA) and t > action and v[DRUGS.index(first_drug)] > 0:
            return float(t), 'post_action_restart'
    return np.inf, 'none'


def clone_censor(t0, action, first_drug, arm, grace_h=12):
    deadline = t0 + grace_h*3600
    own = VA if arm == 1 else NE
    if action <= deadline and first_drug == own:
        return np.inf
    return min(action, deadline)


def weighted_cif(time, event, weight, horizon):
    """Exact-time AJ: event=1 target, 2 competing, 0 censor/horizon."""
    time, event, weight = np.asarray(time), np.asarray(event), np.asarray(weight)
    surv, cif = 1., 0.
    for t in np.unique(time[(event > 0) & (time <= horizon)]):
        risk = weight[time >= t].sum()
        if risk <= 0:
            break
        at_t = time == t
        d1 = weight[at_t & (event == 1)].sum()
        d2 = weight[at_t & (event == 2)].sum()
        cif += surv*d1/risk
        surv *= max(0., 1-(d1+d2)/risk)
    return float(cif)
