"""Read, validate and clean UCI Online Retail II, then build a daily SKU panel.

The raw file is an invoice-line log. The model needs a daily demand series per
SKU, so this module does three distinct jobs and keeps them separable:

    load_raw            -> every line of both sheets, unmodified
    clean_transactions  -> product lines only, with a written reason per rule
    build_daily_panel   -> dense daily demand for the SKUs we actually model

Nothing here imputes a value silently. Every row that disappears is counted
and reported in reports/data_quality_report.json.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.config import ensure_dir, load_config, resolve

# Canonical internal column names. The raw workbook names are mapped onto
# these immediately so the rest of the codebase never sees "Customer ID".
COLS = {
    "invoice": "invoice",
    "stock_code": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoice_date": "invoice_date",
    "price": "price",
    "customer_id": "customer_id",
    "country": "country",
}


@dataclass
class CleaningLog:
    """Row-level accounting for the cleaning pipeline.

    Exists so that 'we removed 23% of rows' is always backed by a breakdown
    rather than a hand-wave.
    """

    rows_in: int = 0
    rows_out: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, before: int, after: int, reason: str) -> None:
        self.steps.append(
            {
                "step": name,
                "rows_before": int(before),
                "rows_after": int(after),
                "rows_removed": int(before - after),
                "pct_removed": round(100.0 * (before - after) / before, 4) if before else 0.0,
                "reason": reason,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_removed_total": self.rows_in - self.rows_out,
            "pct_removed_total": (
                round(100.0 * (self.rows_in - self.rows_out) / self.rows_in, 4)
                if self.rows_in
                else 0.0
            ),
            "steps": self.steps,
        }


# ----------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------


def _stream_sheet(xlsx_path: Path, sheet: str, name_map: dict[str, str], chunk: int):
    """Yield DataFrame chunks from one worksheet without loading the workbook.

    pandas.read_excel materialises the entire sheet (and openpyxl's object
    graph) before returning, which needs several GB for this file. openpyxl's
    read_only iterator streams one row at a time instead, so peak memory stays
    proportional to `chunk` rather than to the file.
    """
    from openpyxl import load_workbook

    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    try:
        if sheet not in wb.sheetnames:
            raise ValueError(
                f"Sheet '{sheet}' not found in workbook. "
                f"Available sheets: {wb.sheetnames}"
            )
        ws = wb[sheet]
        rows = ws.iter_rows(values_only=True)

        try:
            header = [str(h).strip() if h is not None else "" for h in next(rows)]
        except StopIteration:
            raise ValueError(f"Sheet '{sheet}' is empty.") from None

        missing = set(name_map) - set(header)
        if missing:
            raise ValueError(
                f"Sheet '{sheet}' is missing expected columns: {sorted(missing)}.\n"
                f"Found: {header}\n"
                "The workbook layout differs from config/project.yaml."
            )

        buf: list[tuple] = []
        for row in rows:
            buf.append(row)
            if len(buf) >= chunk:
                yield pd.DataFrame(buf, columns=header)
                buf = []
        if buf:
            yield pd.DataFrame(buf, columns=header)
    finally:
        wb.close()


def load_raw(force_reload: bool = False, chunk_rows: int = 50_000) -> pd.DataFrame:
    """Load both sheets of the workbook into one frame, cached as Parquet.

    The .xlsx is streamed to Parquet on first use (a few minutes), after which
    every later call reads the Parquet cache in a second or two. Pass
    force_reload=True to rebuild the cache.
    """
    cfg = load_config()
    xlsx_path = resolve(cfg["paths"]["raw_online_retail"])
    cache_path = resolve(cfg["paths"]["interim"]) / "online_retail_raw.parquet"

    if cache_path.exists() and not force_reload:
        return pd.read_parquet(cache_path)

    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at:\n  {xlsx_path}\n\n"
            "This project does not ship data. Download it first:\n"
            "  python scripts/download_data.py\n\n"
            "See docs/dataset_acquisition.md for details."
        )

    import pyarrow as pa
    import pyarrow.parquet as pq

    name_map = {v: COLS[k] for k, v in cfg["dataset"]["columns"].items()}

    # Fixed schema so every chunk writes compatibly, regardless of what types
    # pandas happens to infer from a given 50k-row slice.
    schema = pa.schema(
        [
            ("invoice", pa.string()),
            ("stock_code", pa.string()),
            ("description", pa.string()),
            ("quantity", pa.float64()),
            ("invoice_date", pa.timestamp("ns")),
            ("price", pa.float64()),
            ("customer_id", pa.float64()),
            ("country", pa.string()),
            ("source_sheet", pa.string()),
        ]
    )

    ensure_dir(cache_path)
    tmp_path = cache_path.with_suffix(".tmp.parquet")
    writer = pq.ParquetWriter(tmp_path, schema, compression="snappy")
    try:
        for sheet in cfg["dataset"]["sheets"]:
            for part in _stream_sheet(xlsx_path, sheet, name_map, chunk_rows):
                part = part.rename(columns=name_map)
                part = part[list(name_map.values())]
                part["source_sheet"] = sheet

                for col in ("invoice", "stock_code", "description", "country"):
                    part[col] = part[col].astype("string").astype(object)
                for col in ("quantity", "price", "customer_id"):
                    part[col] = pd.to_numeric(part[col], errors="coerce")
                part["invoice_date"] = pd.to_datetime(
                    part["invoice_date"], errors="coerce"
                )

                writer.write_table(
                    pa.Table.from_pandas(part, schema=schema, preserve_index=False)
                )
                del part
    finally:
        writer.close()

    tmp_path.replace(cache_path)
    return pd.read_parquet(cache_path)


# ----------------------------------------------------------------------
# 2. Validate
# ----------------------------------------------------------------------


def validate_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Fail loudly if the frame is not what the rest of the pipeline expects."""
    required = set(COLS.values())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after rename: {sorted(missing)}")

    if not pd.api.types.is_datetime64_any_dtype(df["invoice_date"]):
        raise TypeError(
            f"invoice_date must be datetime, got {df['invoice_date'].dtype}. "
            "The Excel reader normally parses this automatically."
        )
    for col in ("quantity", "price"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"{col} must be numeric, got {df[col].dtype}")

    if len(df) == 0:
        raise ValueError("Loaded dataset is empty.")

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "date_min": str(df["invoice_date"].min()),
        "date_max": str(df["invoice_date"].max()),
    }


# ----------------------------------------------------------------------
# 3. Clean
# ----------------------------------------------------------------------


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    """Reduce the raw log to genuine, positive product sales lines.

    Returns the cleaned frame plus a log explaining every removal. Returned
    (cancelled / negative) lines are split out rather than deleted blindly --
    they are summarised and attached as a per-SKU-per-day returns series,
    because a high return rate is itself a demand signal.
    """
    cfg = load_config()["cleaning"]
    log = CleaningLog(rows_in=len(df))
    out = df.copy()

    # -- drop exact duplicate invoice lines -------------------------------
    before = len(out)
    out = out.drop_duplicates()
    log.record(
        "drop_exact_duplicates",
        before,
        len(out),
        "Identical rows on every column are re-scans of the same line, "
        "not two separate sales.",
    )

    # -- rows with no usable date or SKU ----------------------------------
    before = len(out)
    out = out.dropna(subset=["invoice_date", "stock_code"])
    log.record(
        "drop_missing_key_fields",
        before,
        len(out),
        "A line with no date or no stock code cannot enter a time series.",
    )

    # -- normalise the stock code before matching -------------------------
    out["stock_code"] = out["stock_code"].astype(str).str.strip().str.upper()
    out["invoice"] = out["invoice"].astype(str).str.strip()

    # -- separate cancellations -------------------------------------------
    prefix = cfg["cancellation_prefix"]
    is_cancel = out["invoice"].str.startswith(prefix)
    cancellations = out.loc[is_cancel].copy()
    before = len(out)
    out = out.loc[~is_cancel].copy()
    log.record(
        "split_cancellations",
        before,
        len(out),
        f"Invoices prefixed '{prefix}' are credit notes. They are removed from "
        "the demand series but retained separately as a returns signal.",
    )

    # -- keep only genuine product codes ----------------------------------
    pattern = re.compile(cfg["product_code_regex"])
    is_product = out["stock_code"].str.match(pattern)
    removed_codes = (
        out.loc[~is_product, "stock_code"].value_counts().head(25).to_dict()
    )
    before = len(out)
    out = out.loc[is_product].copy()
    log.record(
        "drop_non_product_codes",
        before,
        len(out),
        "Service lines (POST, DOT, M, BANK CHARGES, ADJUST, TEST*) are not "
        "stockable items and must not be forecast or reordered. "
        f"Most frequent removed codes: {removed_codes}",
    )

    # -- non-positive quantity --------------------------------------------
    before = len(out)
    out = out.loc[out["quantity"] > 0].copy()
    log.record(
        "drop_nonpositive_quantity",
        before,
        len(out),
        "Zero or negative quantities on a non-cancellation invoice are "
        "adjustments, not demand.",
    )

    # -- non-positive price -----------------------------------------------
    if cfg.get("drop_nonpositive_price", True):
        before = len(out)
        out = out.loc[out["price"] > 0].copy()
        log.record(
            "drop_nonpositive_price",
            before,
            len(out),
            "Zero-price lines are samples, write-offs or data-entry errors; "
            "they would distort the revenue-based SKU ranking.",
        )

    # -- line-level quantity outliers --------------------------------------
    cap = cfg["max_line_quantity"]
    before = len(out)
    out = out.loc[out["quantity"] <= cap].copy()
    log.record(
        "drop_extreme_line_quantity",
        before,
        len(out),
        f"Single lines above {cap} units are bulk corrections, not retail "
        "demand. Removing them stops one row dominating a whole SKU series.",
    )

    # -- derived fields ----------------------------------------------------
    out["revenue"] = out["quantity"] * out["price"]
    out["date"] = out["invoice_date"].dt.normalize()

    # -- returns series, kept as a real feature ---------------------------
    if len(cancellations):
        cancellations["stock_code"] = (
            cancellations["stock_code"].astype(str).str.strip().str.upper()
        )
        cancellations = cancellations.loc[
            cancellations["stock_code"].str.match(pattern)
        ]
        cancellations["date"] = cancellations["invoice_date"].dt.normalize()
        cancellations["returned_units"] = cancellations["quantity"].abs()

    out.attrs["returns"] = (
        cancellations[["date", "stock_code", "returned_units"]]
        if len(cancellations)
        else pd.DataFrame(columns=["date", "stock_code", "returned_units"])
    )

    log.rows_out = len(out)
    return out, log


# ----------------------------------------------------------------------
# 4. Daily panel
# ----------------------------------------------------------------------


def select_modelled_skus(clean: pd.DataFrame) -> pd.DataFrame:
    """Pick the SKUs that get a dense daily series.

    A dense panel across all ~4,600 SKUs is overwhelmingly structural zeros,
    which makes 'predict 0' a strong baseline and the exercise meaningless.
    We therefore model the busiest SKUs and say so out loud.
    """
    cfg = load_config()["panel"]
    metric = cfg["sku_rank_metric"]
    value_col = "revenue" if metric == "revenue" else "quantity"

    per_sku = (
        clean.groupby("stock_code")
        .agg(
            total_value=(value_col, "sum"),
            total_units=("quantity", "sum"),
            active_days=("date", "nunique"),
            first_sale=("date", "min"),
            last_sale=("date", "max"),
            description=("description", lambda s: s.dropna().mode().iat[0] if s.notna().any() else ""),
            median_price=("price", "median"),
        )
        .reset_index()
    )

    eligible = per_sku.loc[per_sku["active_days"] >= cfg["min_active_days"]].copy()
    eligible = eligible.sort_values("total_value", ascending=False)
    selected = eligible.head(cfg["top_n_skus"]).copy()
    selected["sku_rank"] = range(1, len(selected) + 1)

    if len(selected) == 0:
        raise ValueError(
            "No SKU met the selection criteria. Lower panel.min_active_days "
            "in config/project.yaml."
        )
    return selected.reset_index(drop=True)


def build_daily_panel(clean: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dense daily demand per modelled SKU over the days the business traded.

    Trading-day handling matters here. This retailer does not sell every
    calendar day (there are no Saturday invoices at all, plus the Christmas
    shutdown). Zero-filling those days would invent ~15% fake zero-demand
    observations and bias every model downward. So the calendar is built from
    days on which the business actually transacted.
    """
    cfg = load_config()["panel"]
    selected = select_modelled_skus(clean)
    keep = set(selected["stock_code"])

    subset = clean.loc[clean["stock_code"].isin(keep)].copy()

    # Days the business as a WHOLE traded -- derived from all SKUs, not just
    # the modelled ones, so a quiet day for one SKU is still a real zero.
    open_days = pd.Index(sorted(clean["date"].unique()), name="date")

    daily = (
        subset.groupby(["stock_code", "date"])
        .agg(
            demand=("quantity", "sum"),
            revenue=("revenue", "sum"),
            n_invoices=("invoice", "nunique"),
            avg_price=("price", "mean"),
        )
        .reset_index()
    )

    # Reindex each SKU onto the full trading calendar.
    index = pd.MultiIndex.from_product(
        [sorted(keep), open_days], names=["stock_code", "date"]
    )
    panel = (
        daily.set_index(["stock_code", "date"])
        .reindex(index)
        .reset_index()
    )

    # These zeros are genuine: the shop was open and this SKU sold nothing.
    panel["demand"] = panel["demand"].fillna(0.0)
    panel["revenue"] = panel["revenue"].fillna(0.0)
    panel["n_invoices"] = panel["n_invoices"].fillna(0).astype(int)

    # Price is carried forward, not zero-filled -- an unsold item still has
    # a price. Any remaining gap at the start is back-filled from the SKU's
    # first observed price.
    panel["avg_price"] = (
        panel.groupby("stock_code")["avg_price"].ffill().bfill()
    )
    panel["avg_price"] = panel["avg_price"].fillna(
        panel["stock_code"].map(selected.set_index("stock_code")["median_price"])
    )

    # Attach returns, if any, as a same-day signal.
    returns = clean.attrs.get("returns")
    if returns is not None and len(returns):
        ret = (
            returns.loc[returns["stock_code"].isin(keep)]
            .groupby(["stock_code", "date"])["returned_units"]
            .sum()
            .reset_index()
        )
        panel = panel.merge(ret, on=["stock_code", "date"], how="left")
    if "returned_units" not in panel.columns:
        panel["returned_units"] = 0.0
    panel["returned_units"] = panel["returned_units"].fillna(0.0)

    # Static SKU attributes.
    panel = panel.merge(
        selected[["stock_code", "description", "sku_rank", "median_price"]],
        on="stock_code",
        how="left",
    )

    panel = panel.sort_values(["stock_code", "date"]).reset_index(drop=True)

    calendar_days = (open_days.max() - open_days.min()).days + 1
    meta = {
        "n_skus_modelled": int(panel["stock_code"].nunique()),
        "n_skus_total_in_clean_data": int(clean["stock_code"].nunique()),
        "n_trading_days": int(len(open_days)),
        "n_calendar_days_in_span": int(calendar_days),
        "non_trading_days_excluded": int(calendar_days - len(open_days)),
        "date_min": str(open_days.min().date()),
        "date_max": str(open_days.max().date()),
        "panel_rows": int(len(panel)),
        "share_of_clean_revenue_covered": float(
            subset["revenue"].sum() / clean["revenue"].sum()
        ),
        "share_of_clean_units_covered": float(
            subset["quantity"].sum() / clean["quantity"].sum()
        ),
        "pct_zero_demand_rows": float(100.0 * (panel["demand"] == 0).mean()),
    }
    return panel, meta


def save_panel(panel: pd.DataFrame) -> Path:
    """Persist the daily panel to Parquet and return the path."""
    cfg = load_config()
    out = resolve(cfg["paths"]["processed"]) / "daily_demand_panel.parquet"
    ensure_dir(out)
    panel.to_parquet(out, index=False)
    return out


def load_panel() -> pd.DataFrame:
    """Read back the saved daily panel."""
    cfg = load_config()
    path = resolve(cfg["paths"]["processed"]) / "daily_demand_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed panel not found at {path}.\n"
            "Run: python scripts/prepare_data.py"
        )
    return pd.read_parquet(path)
