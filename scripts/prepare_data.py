"""Stage 1 of the pipeline: raw workbook -> clean -> daily panel -> quality report.

Run from the repository root:
    .venv\\Scripts\\python scripts\\prepare_data.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xaierp.ingest import (  # noqa: E402
    build_daily_panel,
    build_quality_report,
    clean_transactions,
    load_raw,
    validate_schema,
    write_quality_report,
)
from xaierp.ingest.online_retail import save_panel, select_modelled_skus  # noqa: E402
from xaierp.utils.config import ensure_dir, load_config, resolve  # noqa: E402
from xaierp.utils.seeds import set_global_seed  # noqa: E402


def main() -> int:
    cfg = load_config()
    set_global_seed(cfg["project"]["random_seed"])

    t0 = time.time()
    print("[1/5] Loading raw workbook (first run reads 45 MB of Excel, ~1-3 min)...")
    raw = load_raw()
    print(f"      raw rows: {len(raw):,}  cols: {raw.shape[1]}  ({time.time()-t0:.1f}s)")

    print("[2/5] Validating schema...")
    schema = validate_schema(raw)
    print(f"      date range: {schema['date_min']} -> {schema['date_max']}")

    print("[3/5] Cleaning transactions...")
    clean, log = clean_transactions(raw)
    print(
        f"      kept {log.rows_out:,} of {log.rows_in:,} rows "
        f"({100.0 * log.rows_out / log.rows_in:.2f}%)"
    )
    for step in log.steps:
        print(
            f"        - {step['step']:<32} removed {step['rows_removed']:>8,} "
            f"({step['pct_removed']:.2f}%)"
        )

    print("[4/5] Building daily demand panel...")
    panel, meta = build_daily_panel(clean)
    print(
        f"      {meta['n_skus_modelled']} SKUs x {meta['n_trading_days']} trading days "
        f"= {meta['panel_rows']:,} rows"
    )
    print(
        f"      excluded {meta['non_trading_days_excluded']} non-trading calendar days"
    )
    print(
        f"      modelled SKUs cover {meta['share_of_clean_revenue_covered']:.1%} "
        f"of cleaned revenue"
    )
    panel_path = save_panel(panel)
    print(f"      saved -> {panel_path.relative_to(ROOT)}")

    # SKU master table, also used by the dashboard.
    skus = select_modelled_skus(clean)
    sku_path = resolve(cfg["paths"]["processed"]) / "sku_master.parquet"
    ensure_dir(sku_path)
    skus.to_parquet(sku_path, index=False)
    print(f"      saved -> {sku_path.relative_to(ROOT)}")

    print("[5/5] Writing data quality report...")
    report = build_quality_report(raw, clean, log, panel, meta)
    report_path = write_quality_report(report)
    print(f"      saved -> {report_path.relative_to(ROOT)}")

    print(f"\nDone in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
