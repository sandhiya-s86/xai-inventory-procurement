"""Build the SKU x feature matrix that every Unit 3 method runs on.

The daily panel is 200 SKUs x 604 trading days. Clustering that directly
would be clustering *SKU-days*, which is not a useful question. What a buyer
actually wants segmented is the SKU: "which products behave alike, so that
one replenishment rule can cover all of them?" So the panel is collapsed to
one row per SKU with columns describing its demand BEHAVIOUR.

Leakage
-------
These profiles feed a segmentation that is used to set safety stock on the
test period. If a profile were computed over the whole series it would carry
information from the test window into a decision taken before it. So
`build_sku_profiles` takes an explicit `end_date` and every statistic is
computed on data at or before it; scripts/run_unsupervised.py passes the end
of the TRAIN split. tests/test_unit3_unsupervised.py checks that moving the
cutoff actually changes the profiles, i.e. that the argument is not ignored.

Column meanings
---------------
    mean_demand       average units/day -- the volume axis
    std_demand        day-to-day variability in units
    cv_demand         std/mean -- variability relative to size, so a big
                      steady SKU and a small steady SKU look alike
    zero_rate         fraction of trading days with no sales at all
    adi               average demand interval = days / selling-days. The
                      standard intermittency measure: 1.0 sells daily, 4.0
                      sells about weekly
    spikiness         p95 / mean -- how extreme the busy days are
    trend_slope       OLS slope of demand on time, divided by mean, so it
                      reads as fractional growth per day
    weekend_ratio     mean weekend demand / mean weekday demand
    q4_ratio          mean Oct-Dec demand / mean rest-of-year demand; this is
                      a gift retailer, so it is the seasonality axis
    acf_lag7          lag-7 autocorrelation -- strength of the weekly cycle
    return_rate       returned units / units sold
    median_price      unit price, the only non-demand column
    revenue_share     share of total revenue, i.e. commercial importance
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b not in (0, 0.0) and np.isfinite(b) else default


def _trend_slope(y: np.ndarray) -> float:
    """OLS slope of y on 0..n-1, normalised by the mean level."""
    n = len(y)
    if n < 3:
        return 0.0
    t = np.arange(n, dtype=float)
    t -= t.mean()
    denom = float(np.sum(t**2))
    if denom == 0:
        return 0.0
    slope = float(np.sum(t * (y - y.mean())) / denom)
    return _safe_div(slope, y.mean())


def _acf(y: np.ndarray, lag: int) -> float:
    """Autocorrelation at a given lag; 0 when the series is constant."""
    if len(y) <= lag + 2:
        return 0.0
    a, b = y[:-lag], y[lag:]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


PROFILE_COLUMNS = [
    "mean_demand",
    "std_demand",
    "cv_demand",
    "zero_rate",
    "adi",
    "spikiness",
    "trend_slope",
    "weekend_ratio",
    "q4_ratio",
    "acf_lag7",
    "return_rate",
    "median_price",
    "revenue_share",
]


def build_sku_profiles(
    panel: pd.DataFrame, end_date: str | pd.Timestamp | None = None
) -> pd.DataFrame:
    """One behavioural profile row per SKU, using data up to `end_date` only.

    Returns a frame indexed by `stock_code` with `PROFILE_COLUMNS` plus the
    description, ready to be standardised and handed to PCA / FA / clustering.
    """
    df = panel.copy()
    if end_date is not None:
        df = df.loc[df["date"] <= pd.Timestamp(end_date)]
    if df.empty:
        raise ValueError(f"No panel rows at or before end_date={end_date!r}")

    df = df.sort_values(["stock_code", "date"])
    total_revenue = float(df["revenue"].sum())

    rows = []
    for sku, g in df.groupby("stock_code", sort=True):
        d = g["demand"].to_numpy(dtype=float)
        n = len(d)
        mean = float(d.mean())
        nonzero = int((d > 0).sum())

        is_weekend = g["date"].dt.dayofweek >= 5
        is_q4 = g["date"].dt.month >= 10

        wk_end = d[is_weekend.to_numpy()]
        wk_day = d[~is_weekend.to_numpy()]
        q4 = d[is_q4.to_numpy()]
        rest = d[~is_q4.to_numpy()]

        rows.append(
            {
                "stock_code": sku,
                "description": str(g["description"].iloc[0]),
                "mean_demand": mean,
                "std_demand": float(d.std(ddof=0)),
                "cv_demand": _safe_div(float(d.std(ddof=0)), mean),
                "zero_rate": float((d == 0).mean()),
                # n/nonzero: 1.0 = sells every trading day. Capped at the
                # series length when a SKU never sold in the window.
                "adi": _safe_div(n, nonzero, default=float(n)),
                "spikiness": _safe_div(float(np.percentile(d, 95)), mean),
                "trend_slope": _trend_slope(d),
                "weekend_ratio": _safe_div(
                    float(wk_end.mean()) if len(wk_end) else 0.0,
                    float(wk_day.mean()) if len(wk_day) else 0.0,
                ),
                "q4_ratio": _safe_div(
                    float(q4.mean()) if len(q4) else 0.0,
                    float(rest.mean()) if len(rest) else 0.0,
                ),
                "acf_lag7": _acf(d, 7),
                "return_rate": _safe_div(
                    float(g["returned_units"].sum()), float(d.sum())
                ),
                "median_price": float(g["median_price"].iloc[0]),
                "revenue_share": _safe_div(float(g["revenue"].sum()), total_revenue),
                "n_days": n,
            }
        )

    out = pd.DataFrame(rows).set_index("stock_code")
    # Any remaining non-finite value is a degenerate SKU (e.g. no weekday
    # sales at all); 0 is the neutral value on every one of these ratios.
    out[PROFILE_COLUMNS] = (
        out[PROFILE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    return out


def profile_matrix(profiles: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Numeric matrix plus the column names, in a fixed order."""
    return profiles[PROFILE_COLUMNS].to_numpy(dtype=float), list(PROFILE_COLUMNS)


def segment_summary(profiles: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean profile of each cluster, for naming and interpreting the segments."""
    out = profiles.copy()
    out["segment"] = labels
    agg = out.groupby("segment")[PROFILE_COLUMNS].mean()
    agg.insert(0, "n_skus", out.groupby("segment").size())
    agg["revenue_share_total"] = out.groupby("segment")["revenue_share"].sum()
    return agg.reset_index()


# The behavioural axis each profile column speaks to, and what a HIGH or LOW
# value on it means in plain words. Only columns a buyer would recognise.
_AXIS_WORDS = {
    "mean_demand": ("fast-moving", "slow-moving"),
    "cv_demand": ("erratic", "steady"),
    "zero_rate": ("sporadic", "sells-daily"),
    "adi": ("intermittent", "continuous"),
    "spikiness": ("spiky", "flat-profile"),
    "q4_ratio": ("Christmas-peaking", "non-seasonal"),
    "acf_lag7": ("weekly-cyclical", "acyclical"),
    "trend_slope": ("growing", "declining"),
    "median_price": ("premium-priced", "low-priced"),
    "revenue_share": ("revenue-critical", "long-tail"),
    "return_rate": ("return-prone", "rarely-returned"),
}


def name_segments(summary: pd.DataFrame) -> dict[int, str]:
    """Name each segment by what most DISTINGUISHES it from the others.

    Fixed thresholds ("high volume = more than 50/day") stop meaning anything
    the moment the data changes, and naming every segment off the same two
    columns collapses several of them onto the same caption. So instead each
    segment's mean profile is z-scored against the other SEGMENT means, and
    the segment is named after the axis on which it is most extreme -- the
    answer to "what is unusual about this group?".

    A volume tier is prefixed for orientation, since that is the first thing
    a buyer wants to know. Names are a reading aid for the dashboard; nothing
    downstream depends on the string.
    """
    axes = [c for c in _AXIS_WORDS if c in summary.columns]
    block = summary[axes].astype(float)

    # z-score each axis ACROSS segments, so "extreme" means extreme relative
    # to the other segments rather than to the raw units.
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=0).replace(0.0, np.nan)
    Zs = ((block - mu) / sd).fillna(0.0)

    vol_rank = summary["mean_demand"].rank(pct=True)

    names: dict[int, str] = {}
    used: set[str] = set()

    for i, row in summary.iterrows():
        seg = int(row["segment"])

        # A segment with no sales at all in the window is its own category:
        # every ratio in its profile is a divide-by-zero fallback, so the
        # z-scores below would be describing placeholder values.
        if float(row["mean_demand"]) <= 0.01:
            names[seg] = "Dormant (no sales in window)"
            continue

        tier = (
            "High-volume" if vol_rank.loc[i] >= 2 / 3
            else "Mid-volume" if vol_rank.loc[i] >= 1 / 3
            else "Low-volume"
        )

        # Walk the axes most-extreme-first and take the first descriptor not
        # already claimed, so two segments never end up with the same caption.
        ranked = Zs.loc[i].abs().sort_values(ascending=False)
        descriptor = None
        for axis in ranked.index:
            hi, lo = _AXIS_WORDS[axis]
            word = hi if Zs.loc[i, axis] >= 0 else lo
            if word not in used:
                used.add(word)
                descriptor = word
                break
        if descriptor is None:                       # pragma: no cover
            descriptor = "mixed"

        names[seg] = f"{tier} {descriptor}"

    # Final guard: identical captions can still arise from the tier prefix.
    seen: dict[str, int] = {}
    for seg, label in list(names.items()):
        if label in seen:
            seen[label] += 1
            names[seg] = f"{label} ({seen[label]})"
        else:
            seen[label] = 1
    return names
