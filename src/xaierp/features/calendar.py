"""Calendar features.

These are the only features that are trivially safe: a date's weekday is
knowable arbitrarily far in advance, so no leakage is possible.

Online Retail II has no holiday table, so UK bank holidays relevant to the
2009-2011 span are encoded explicitly rather than guessed. Dates are from the
UK government's published bank-holiday record for England and Wales.
"""

from __future__ import annotations

import pandas as pd

# England & Wales bank holidays covering the dataset span (2009-12 -> 2011-12).
UK_BANK_HOLIDAYS = pd.to_datetime(
    [
        "2009-12-25", "2009-12-28",
        "2010-01-01", "2010-04-02", "2010-04-05", "2010-05-03",
        "2010-05-31", "2010-08-30", "2010-12-27", "2010-12-28",
        "2011-01-03", "2011-04-22", "2011-04-25", "2011-04-29",
        "2011-05-02", "2011-05-30", "2011-08-29",
    ]
)


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Attach day/week/month/quarter structure and holiday proximity."""
    out = df.copy()
    d = out[date_col]

    out["day_of_week"] = d.dt.dayofweek.astype("int16")
    out["day_of_month"] = d.dt.day.astype("int16")
    out["day_of_year"] = d.dt.dayofyear.astype("int16")
    out["week_of_year"] = d.dt.isocalendar().week.astype("int16")
    out["month"] = d.dt.month.astype("int16")
    out["quarter"] = d.dt.quarter.astype("int16")
    out["year"] = d.dt.year.astype("int16")

    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    out["is_month_start"] = d.dt.is_month_start.astype("int8")
    out["is_month_end"] = d.dt.is_month_end.astype("int8")
    out["is_quarter_end"] = d.dt.is_quarter_end.astype("int8")

    out["is_bank_holiday"] = d.isin(UK_BANK_HOLIDAYS).astype("int8")

    # Days to the next bank holiday -- a genuine forward-looking but
    # legitimately known-in-advance signal (the calendar is public).
    hol = UK_BANK_HOLIDAYS.sort_values()
    idx = hol.searchsorted(d.values, side="left")
    idx = idx.clip(0, len(hol) - 1)
    out["days_to_bank_holiday"] = (
        (hol[idx].values - d.values).astype("timedelta64[D]").astype("int32")
    ).clip(0, 120)

    # Pre-Christmas trading is the dominant seasonal effect for this retailer.
    out["is_q4_peak"] = ((out["month"] == 11) | (out["month"] == 12)).astype("int8")

    return out
