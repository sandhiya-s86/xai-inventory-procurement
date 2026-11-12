"""Drive the Streamlit dashboard with Playwright and verify every page renders.

A dashboard that imports cleanly is not a dashboard that works. Streamlit
catches exceptions per-widget and renders them as a red box in the page, so a
broken chart or a missing column shows up as a running app with an error in
it -- which a unit test of the module will never see. This script launches
the real server, clicks through all eight pages in a real browser, and fails
if any page shows a Streamlit exception or is missing the content it should
have.

    .venv\\Scripts\\python scripts\\verify_dashboard.py
    .venv\\Scripts\\python scripts\\verify_dashboard.py --headed   # watch it

Writes:
    reports/dashboard_verification.json
    reports/screenshots/page_*.png
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PAGES = [
    # (sidebar label, substrings that MUST appear once the page has rendered)
    ("1 · Overview", ["Overview", "Total daily demand", "SKUs modelled"]),
    ("2 · Demand Prediction", ["Demand Prediction", "Forecast for the 7 days"]),
    ("3 · Inventory Risk", ["Inventory Risk", "risk SKUs"]),
    ("4 · Procurement Recommendation",
     ["Procurement Recommendation", "SKUs to reorder", "Policy comparison"]),
    ("5 · WHY?  (Explainability)",
     ["Explaining the recommendation", "Quantity decomposition", "What if"]),
    ("6 · Model Comparison",
     ["Model Comparison", "Regression", "Classification", "Polynomial degree study"]),
    # The two new Unit 3 / Unit 4 pages.
    ("7 · SKU Segmentation (Unit 3)",
     ["SKU Segmentation", "Segment profiles", "SKUs segmented",
      "Hard vs soft assignment", "Choosing the number of latent dimensions"]),
    ("8 · Demand Regimes (Unit 4)",
     ["Demand Regimes", "The three learned regimes", "Why this matters for stocking",
      "Kalman filter", "Two decoders", "Markov model of demand states"]),
]

# Streamlit renders a tracebacks in an element with this test id; its presence
# is the single most reliable signal that a page is broken.
EXCEPTION_SELECTOR = '[data-testid="stException"]'
ALERT_ERROR_SELECTOR = '[data-baseweb="notification"][kind="error"]'


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_server(port: int, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headed", action="store_true",
                    help="show the browser instead of running headless")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--slow", type=int, default=0,
                    help="slow_mo milliseconds, for watching it work")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed. Run:")
        print("  .venv\\Scripts\\python -m pip install playwright")
        print("  .venv\\Scripts\\python -m playwright install chromium")
        return 2

    port = args.port or free_port()
    shots = ROOT / "reports" / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("DASHBOARD VERIFICATION (Playwright)")
    print("=" * 72)
    print(f"Starting Streamlit on port {port} ...")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "none",
            "--logger.level", "error",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    results: list[dict] = []
    ok_all = True

    try:
        if not wait_for_server(port):
            print("FAILED: Streamlit did not start within 120s")
            out = proc.stdout.read() if proc.stdout else ""
            print(out[-4000:])
            return 1
        print("Server is up.\n")

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not args.headed,
                                         slow_mo=args.slow)
            page = browser.new_page(viewport={"width": 1600, "height": 1200})

            # Surface browser-side errors too -- a JS exception can blank a
            # chart while the Python side stays perfectly happy.
            console_errors: list[str] = []
            page.on("console", lambda m: console_errors.append(m.text)
                    if m.type == "error" else None)

            page.goto(f"http://127.0.0.1:{port}", wait_until="domcontentloaded")
            # The app is only ready once the sidebar radio exists.
            page.wait_for_selector('[data-testid="stSidebar"]', timeout=90_000)
            page.wait_for_timeout(4000)

            for idx, (label, expected) in enumerate(PAGES, start=1):
                print(f"[{idx}/{len(PAGES)}] {label}")
                before = len(console_errors)

                # Select the page by its radio label.
                radio = page.get_by_role("radio", name=label, exact=False)
                try:
                    radio.first.click(timeout=20_000)
                except Exception:
                    # Fall back to clicking the visible text.
                    page.get_by_text(label, exact=False).first.click(timeout=20_000)

                # Wait for Streamlit to finish the rerun. The "Running" status
                # widget disappearing is the signal; the timeout is generous
                # because page 5 fits a SHAP explainer on first load.
                page.wait_for_timeout(1500)
                try:
                    page.wait_for_selector('[data-testid="stStatusWidget"]',
                                           state="detached", timeout=90_000)
                except Exception:
                    pass
                page.wait_for_timeout(2500)

                body = page.inner_text("body")

                # 1. Did Streamlit raise?
                n_exc = page.locator(EXCEPTION_SELECTOR).count()
                exc_text = ""
                if n_exc:
                    exc_text = page.locator(EXCEPTION_SELECTOR).first.inner_text()[:800]

                # 2. Did a guard() fire, i.e. a missing artefact?
                n_err_alert = page.locator(ALERT_ERROR_SELECTOR).count()

                shot = shots / f"page_{idx}.png"
                page.screenshot(path=str(shot), full_page=True)

                # Streamlit scrolls an inner container, so `full_page` only
                # ever captures the first viewport. Anything below the fold --
                # which on pages 7 and 8 is most of the figures -- would go
                # unverified. So scroll to the bottom, re-check for exceptions
                # that only appear once a lazy widget renders, and capture it.
                # Streamlit scrolls `section.main` (or the app view container),
                # not the window, so page.mouse.wheel moves nothing. Scroll
                # whichever element actually overflows.
                scroll_js = """(dy) => {
                    const cands = [
                        document.querySelector('section.main'),
                        document.querySelector('[data-testid="stAppViewContainer"]'),
                        document.querySelector('[data-testid="stMain"]'),
                        document.scrollingElement,
                    ].filter(Boolean);
                    for (const el of cands) {
                        if (el.scrollHeight > el.clientHeight + 10) {
                            el.scrollTop = dy < 0 ? 0 : el.scrollHeight;
                            return [el.tagName, el.scrollTop, el.scrollHeight];
                        }
                    }
                    window.scrollTo(0, dy < 0 ? 0 : 100000);
                    return ['window', window.scrollY, document.body.scrollHeight];
                }"""
                page.evaluate(scroll_js, 1)
                page.wait_for_timeout(2500)
                n_exc_bottom = page.locator(EXCEPTION_SELECTOR).count()
                n_err_bottom = page.locator(ALERT_ERROR_SELECTOR).count()
                body += "\n" + page.inner_text("body")
                page.screenshot(path=str(shots / f"page_{idx}_bottom.png"))
                page.evaluate(scroll_js, -1)
                page.wait_for_timeout(600)

                if n_exc_bottom > n_exc:
                    exc_text += (" | below fold: "
                                 + page.locator(EXCEPTION_SELECTOR).last.inner_text()[:400])
                n_exc = max(n_exc, n_exc_bottom)
                n_err_alert = max(n_err_alert, n_err_bottom)

                # 3. Is the content we expect actually on the page? Checked
                #    against the combined top+bottom text, so a heading below
                #    the fold still counts as rendered.
                missing = [s for s in expected if s.lower() not in body.lower()]

                # 4. Any new browser console errors?
                new_console = [
                    e for e in console_errors[before:]
                    # Favicon/websocket noise is not a page defect.
                    if "favicon" not in e.lower() and "websocket" not in e.lower()
                ]

                # 5. Count the widgets that actually drew. A page with no
                #    exceptions but no charts either is still broken.
                n_charts = (
                    page.locator('[data-testid="stVegaLiteChart"]').count()
                    + page.locator("canvas").count()
                )
                n_tables = page.locator('[data-testid="stDataFrame"]').count()
                n_images = page.locator('[data-testid="stImage"] img').count()

                page_ok = (n_exc == 0 and n_err_alert == 0 and not missing)
                ok_all = ok_all and page_ok

                results.append(
                    {
                        "page": label,
                        "ok": page_ok,
                        "n_streamlit_exceptions": n_exc,
                        "exception_text": exc_text,
                        "n_error_alerts": n_err_alert,
                        "missing_expected_text": missing,
                        "console_errors": new_console[:5],
                        "body_chars": len(body),
                        "n_charts": n_charts,
                        "n_tables": n_tables,
                        "n_images": n_images,
                        "screenshot": str(shot.relative_to(ROOT)).replace("\\", "/"),
                    }
                )

                status = "PASS" if page_ok else "FAIL"
                print(f"        {status}  "
                      f"{len(body):,} chars, {n_tables} tables, "
                      f"{n_charts} charts, {n_images} figures, "
                      f"{n_exc} exceptions, {n_err_alert} error alerts")
                if missing:
                    print(f"        MISSING TEXT: {missing}")
                if exc_text:
                    print(f"        EXCEPTION: {exc_text[:300]}")
                if new_console:
                    print(f"        console: {new_console[:2]}")

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:      # pragma: no cover
            proc.kill()

    out = {
        "all_pages_passed": ok_all,
        "n_pages": len(PAGES),
        "n_passed": sum(r["ok"] for r in results),
        "pages": results,
    }
    path = ROOT / "reports" / "dashboard_verification.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"{out['n_passed']}/{out['n_pages']} pages passed")
    print(f"screenshots -> reports/screenshots/")
    print(f"report      -> reports/dashboard_verification.json")
    print("=" * 72)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
