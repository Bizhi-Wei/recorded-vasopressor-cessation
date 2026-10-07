"""Measurement-only cessation variants; original dose arrays are never mutated."""
import numpy as np
import pandas as pd
from rebuild_v3_core import DRUGS, EPS, at, seconds, normalize_rate

STATUSES = ['ChangeDose/Rate', 'Changed', 'FinishedRunning', 'Stopped', 'Paused']

def candidates(tl, start, end):
    # Match core.first_action exactly. At shifted clinical epoch magnitudes,
    # EPS is below floating-point resolution and the end boundary is exclusive.
    for t in tl['bounds'][(tl['bounds'] >= start) & (tl['bounds'] < min(end, tl['bounds'][-1]) + EPS)]:
        if t >= tl['bounds'][-1]:
            continue
        before, after = at(tl, t, True)[:2], at(tl, t)[:2]
        if np.isfinite(before).all() and (before > 0).all() and np.isfinite(after).all():
            stopped = np.flatnonzero(after == 0)
            if len(stopped):
                yield float(t), ('simultaneous' if len(stopped) == 2 else DRUGS[stopped[0]]), stopped

def prepare_rows(rows):
    q = rows.copy()
    q['s'] = pd.to_datetime(q.starttime).to_numpy(dtype='datetime64[ns]').astype('int64') / 1e9
    q['e'] = pd.to_datetime(q.endtime).to_numpy(dtype='datetime64[ns]').astype('int64') / 1e9
    return q[q.statusdescription.isin(STATUSES) & (q.e > q.s)]

def ending_source(prepared, drug, t, expected):
    active = prepared[(prepared.drug == drug) & (prepared.s < t) & (prepared.e >= t)]
    active = active.sort_values(['s', 'rate'], kind='stable')
    if active.empty:
        raise AssertionError('No active source row at cessation')
    winner = active.iloc[-1]
    tied = active[(active.s == winner.s) & (active.rate == winner.rate)]
    value = normalize_rate(drug, winner.rate, winner.rateuom, winner.patientweight)
    assert winner.e == t and np.isfinite(value) and value > 0 and np.isclose(value, expected, rtol=1e-12, atol=1e-12)
    return dict(drug=drug, status=str(winner.statusdescription), paused=bool(tied.statusdescription.eq('Paused').any()),
                tied_rows=len(tied), tied_statuses=';'.join(sorted(tied.statusdescription.unique())),
                status_tie_disagreement=tied.statusdescription.nunique() > 1,
                orderid=int(winner.orderid), linkorderid=int(winner.linkorderid),
                original_start=winner.starttime, source_end=winner.endtime,
                raw_rate=float(winner.rate), rateuom=winner.rateuom, normalized_rate=float(value))

def bridge_gaps(tl, minutes, stable, deadline, observation_end):
    """Bridge only original finite zero runs; leave dose arrays intact.

    Virtual rates contain 1 for positive delivery and 0/NaN otherwise. A 1
    represents continuity of a classification, never an imputed dose.
    """
    b, v = tl['bounds'], tl['rates']
    virtual = np.where(np.isnan(v), np.nan, (v > 0).astype(float))
    gaps, terminal = [], []
    if minutes == 0:
        return {'bounds': b.copy(), 'rates': virtual}, gaps, terminal
    limit = minutes * 60
    for j in (0, 1):
        i = 0
        while i < len(v):
            if v[i, j] != 0:
                i += 1
                continue
            k = i + 1
            while k < len(v) and v[k, j] == 0:
                k += 1
            a, z = float(b[i]), float(b[k])
            left = i > 0 and np.isfinite(v[i-1, j]) and v[i-1, j] > 0
            right = k < len(v) and np.isfinite(v[k, j]) and v[k, j] > 0
            relevant = a >= stable and a <= deadline + EPS and a < observation_end
            if left and right and z-a <= limit + EPS and z < observation_end:
                virtual[i:k, j] = 1
                if relevant:
                    gaps.append(dict(drug=DRUGS[j], gap_start=a, restart=z, gap_minutes=(z-a)/60,
                                     restart_after_deadline=z > deadline, threshold_minutes=minutes))
            elif left and relevant and observation_end-a < limit and z >= observation_end:
                terminal.append(dict(drug=DRUGS[j], gap_start=a, observed_gap_minutes=(observation_end-a)/60,
                                     threshold_minutes=minutes, status='SHORT_TERMINAL_GAP_PERSISTENCE_UNRESOLVED'))
            i = k
    return {'bounds': b.copy(), 'rates': virtual}, gaps, terminal
