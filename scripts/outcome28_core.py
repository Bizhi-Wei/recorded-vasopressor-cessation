"""Pure interval rules for recorded 28-day outcome observability."""
import pandas as pd

def death_status(t0, end, exact, dod, coverage_end, conflict=False):
    if conflict:
        return 'RECORD_CONFLICT'
    if pd.notna(exact):
        if exact <= t0:
            return 'DEATH_AT_OR_BEFORE_ANCHOR'
        return 'DEATH_WITHIN28_EXACT' if exact <= end else 'NO_DEATH_WITHIN28_LATER_DEATH'
    if pd.notna(dod):
        lo, hi = dod.normalize(), dod.normalize() + pd.Timedelta(days=1)
        if hi <= t0:
            return 'DEATH_AT_OR_BEFORE_ANCHOR'
        if lo <= t0 < hi:
            return 'DATE_AMBIGUOUS_AT_ANCHOR'
        if hi <= end:
            return 'DEATH_WITHIN28_DATE'
        if lo <= end < hi:
            return 'DATE_AMBIGUOUS_AT_DAY28'
        return 'NO_DEATH_WITHIN28_LATER_DEATH'
    return ('NO_DEATH_RECORDED_WITHIN_LINKAGE_WINDOW' if pd.notna(coverage_end) and end <= coverage_end
            else 'LINKAGE_WINDOW_INSUFFICIENT')

def union_seconds(intervals, start, end):
    clipped = sorted((max(a, start), min(b, end)) for a,b in intervals
                     if a is not None and b is not None and b > start and a < end and b > a)
    total, current = 0., None
    for a,b in clipped:
        if current is None:
            current = (a,b)
        elif a <= current[1]:
            current = (current[0], max(current[1],b))
        else:
            total += current[1]-current[0]
            current = (a,b)
    return total + (current[1]-current[0] if current else 0.)
