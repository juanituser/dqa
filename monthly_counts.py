import sys
from collections import defaultdict
import numpy as np
import pandas as pd
import osmium

ROADS = {"trunk","trunk_link","motorway","motorway_link","primary","primary_link",
         "secondary","secondary_link","tertiary","tertiary_link","footway",
         "living_street","unclassified","residential","pedestrian","service","track"}
POI = {"hospital","clinic","place_of_worship"}

def classify(tags):
    """Return the feature class of one VERSION, or None if it is not in scope."""
    if tags.get("highway") in ROADS:
        return "road"
    if "building" in tags:
        return "building"
    if tags.get("amenity") in POI:
        return tags["amenity"]
    return None

def versions(path):
    """Yield (key, timestamp, active_class) for every version, in file order."""
    for o in osmium.FileProcessor(path):          # reads ALL versions (history file)
        kind = "n" if o.is_node() else "w" if o.is_way() else "r"
        cls = classify(o.tags) if o.visible else None   # deleted version -> None
        yield (kind, o.id), o.timestamp, cls

def monthly_counts(path, start="2008-01", end=None):
    months = pd.period_range(start, end or pd.Timestamp.now("UTC").strftime("%Y-%m"), freq="M")
    idx = {m: i for i, m in enumerate(months)}
    diff = defaultdict(lambda: np.zeros(len(months) + 1, dtype=int))

    def month_i(ts):                                # month the event falls in (clamped)
        p = pd.Period(ts.strftime("%Y-%m"), freq="M")
        return min(max(idx.get(p, 0 if p < months[0] else len(months)), 0), len(months))

    prev = None                                      # (key, ts, cls) of previous version
    def close(prev, end_ts):
        key, ts, cls = prev
        if cls is not None:
            a = month_i(ts)
            b = len(months) if end_ts is None else month_i(end_ts)
            if b > a:
                diff[cls][a] += 1
                diff[cls][b] -= 1

    for key, ts, cls in versions(path):
        if prev is not None:
            close(prev, ts if prev[0] == key else None)   # same object -> ends at next version
        prev = (key, ts, cls)
    if prev is not None:
        close(prev, None)

    out = pd.DataFrame({c: np.cumsum(d)[:-1] for c, d in diff.items()}, index=months.to_timestamp("M"))
    return out

if __name__ == "__main__":
    df = monthly_counts(sys.argv[1])
    df.to_csv(sys.argv[2])
    print(df.loc[["2009-10-31","2009-11-30","2010-04-30","2010-06-30","2011-08-31","2012-12-31","2013-01-31","2014-01-31","2015-01-31","2015-02-28"]])
