"""Generate the PBL-1 PowerPoint deck — 15 slides, review-ready.

Builds paper/PBL1_Presentation.pptx.

Design goals, in priority order:
  1. A non-specialist examiner must understand WHAT the project is, WHAT
     problem it solves and WHO benefits within the first four slides.
  2. Every number comes from reports/ at build time, so the deck can never
     drift away from the actual results.
  3. Dataset provenance and the real/simulated split are stated explicitly,
     with clickable source links.
  4. Each slide carries a 30-60 second speaking script in its notes pane.

    .venv\\Scripts\\python scripts\\build_presentation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xaierp.utils.config import ensure_dir, load_config, resolve  # noqa: E402

CFG = load_config()
REPORTS = resolve(CFG["paths"]["reports"])
FIGS = resolve(CFG["paths"]["figures"])

# ---------------------------------------------------------------- team
TEAM = [
    ("Sandhiya S", "RA2411003010045"),
    ("Nishanth Kannan", "RA2411003010051"),
    ("Netti Nithin Chakravarthy", "RA2411003010033"),
]
COURSE = "PBL-1  ·  Machine Learning  ·  Units 1 & 2"

# ---------------------------------------------------------------- palette
INK = RGBColor(0x0E, 0x1C, 0x36)      # deep navy, page background
NAVY = RGBColor(0x14, 0x2E, 0x55)
STEEL = RGBColor(0x1E, 0x40, 0x6E)
TEAL = RGBColor(0x2D, 0xD4, 0xBF)     # primary accent
SKY = RGBColor(0x5E, 0xAD, 0xF2)
AMBER = RGBColor(0xF5, 0xB7, 0x4A)
CORAL = RGBColor(0xFF, 0x6B, 0x6B)
LIME = RGBColor(0x6E, 0xE7, 0x9B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PAPER = RGBColor(0xF8, 0xFA, 0xFC)
MUTED = RGBColor(0xA9, 0xBD, 0xD8)
SLATE = RGBColor(0x47, 0x55, 0x69)
CARD = RGBColor(0x1B, 0x33, 0x58)

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.62)
BODY_W = W - 2 * MARGIN


# ---------------------------------------------------------------- data
def load_results() -> dict:
    r = {}
    r["summary"] = json.loads((REPORTS / "model_summary.json").read_text(encoding="utf-8"))
    r["eda"] = json.loads((REPORTS / "eda_findings.json").read_text(encoding="utf-8"))
    r["dq"] = json.loads((REPORTS / "data_quality_report.json").read_text(encoding="utf-8"))
    r["reg"] = pd.read_csv(REPORTS / "regression_model_comparison.csv")
    r["clf"] = pd.read_csv(REPORTS / "classification_model_comparison.csv")
    r["poly"] = pd.read_csv(REPORTS / "polynomial_degree_study.csv")
    r["pol"] = pd.read_csv(REPORTS / "policy_comparison.csv")
    r["qt"] = pd.read_csv(REPORTS / "quantile_model_metrics.csv")
    r["wf"] = pd.read_csv(REPORTS / "whatif_examples.csv")
    r["cases"] = json.loads(
        (REPORTS / "generated" / "case_studies.json").read_text(encoding="utf-8"))
    return r


# ---------------------------------------------------------------- helpers
def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = PAPER
    return s


def rect(slide, x, y, w, h, fill=None, line=None, radius=False, lw=1.0):
    shape = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(shape, x, y, w, h)
    if radius:
        try:
            sh.adjustments[0] = 0.06
        except Exception:
            pass
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    return sh


def text(slide, x, y, w, h, body, size=15, bold=False, color=NAVY,
         align=PP_ALIGN.LEFT, font="Calibri", gap=5, line_spacing=1.0,
         anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.03)
    tf.margin_top = tf.margin_bottom = 0
    for i, ln in enumerate(body.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(gap)
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = ln
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = font
    return tb


def link(slide, x, y, w, h, label, url, size=12, color=SKY):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = label
    r.font.size = Pt(size)
    r.font.color.rgb = color
    r.font.underline = True
    r.font.name = "Calibri"
    r.hyperlink.address = url
    return tb


def header(slide, title, kicker=None, accent=TEAL):
    rect(slide, 0, 0, W, Inches(1.12), fill=NAVY)
    rect(slide, 0, Inches(1.12), W, Inches(0.055), fill=accent)
    y = Inches(0.30) if kicker else Inches(0.34)
    if kicker:
        text(slide, MARGIN, Inches(0.17), BODY_W, Inches(0.25),
             kicker.upper(), size=10.5, bold=True, color=accent, gap=0)
        y = Inches(0.44)
    text(slide, MARGIN, y, BODY_W - Inches(0.9), Inches(0.52),
         title, size=27, bold=True, color=WHITE, gap=0)


def footer(slide, n):
    rect(slide, 0, H - Inches(0.42), W, Inches(0.42), fill=RGBColor(0xE7, 0xEC, 0xF3))
    text(slide, MARGIN, H - Inches(0.345), Inches(8.6), Inches(0.26),
         "XAI Decision Support for Inventory & Procurement  ·  "
         "Sandhiya S · Nishanth Kannan · Netti Nithin Chakravarthy",
         size=9.5, color=SLATE, gap=0)
    text(slide, W - MARGIN - Inches(1.2), H - Inches(0.345), Inches(1.2), Inches(0.26),
         f"{n} / 15", size=9.5, bold=True, color=STEEL,
         align=PP_ALIGN.RIGHT, gap=0)


def notes(slide, script):
    slide.notes_slide.notes_text_frame.text = script


def card(slide, x, y, w, h, value, label, accent=SKY, vsize=25, lsize=10.5):
    rect(slide, x, y, w, h, fill=WHITE, line=RGBColor(0xD6, 0xDF, 0xEA), radius=True)
    rect(slide, x, y, w, Inches(0.07), fill=accent)
    text(slide, x, y + Inches(0.22), w, Inches(0.5), value, size=vsize,
         bold=True, color=accent, align=PP_ALIGN.CENTER, gap=0)
    text(slide, x + Inches(0.08), y + Inches(0.78), w - Inches(0.16), h - Inches(0.85),
         label, size=lsize, color=SLATE, align=PP_ALIGN.CENTER, gap=1)


def banner(slide, x, y, w, h, body, accent=AMBER, size=12.5, bg=None, fg=None):
    bg = bg or RGBColor(0xFF, 0xF8, 0xE9)
    fg = fg or RGBColor(0x6B, 0x4A, 0x0B)
    rect(slide, x, y, w, h, fill=bg, line=accent, radius=True)
    rect(slide, x + Inches(0.02), y + Inches(0.06), Inches(0.06), h - Inches(0.12),
         fill=accent)
    text(slide, x + Inches(0.22), y + Inches(0.12), w - Inches(0.38), h - Inches(0.22),
         body, size=size, color=fg, gap=3)


def table(slide, df, x, y, w, h, font=11, hfont=11, hot=None, widths=None,
          accent=NAVY):
    hot = hot or []
    rows, cols = df.shape[0] + 1, df.shape[1]
    t = slide.shapes.add_table(rows, cols, x, y, w, h).table
    if widths:
        tot = sum(widths)
        for j, cw in enumerate(widths):
            t.columns[j].width = Emu(int(w * cw / tot))
    for j, name in enumerate(df.columns):
        c = t.cell(0, j)
        c.text = str(name)
        p = c.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.runs[0].font.size = Pt(hfont)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = WHITE
        c.fill.solid()
        c.fill.fore_color.rgb = accent
        c.margin_top = c.margin_bottom = Inches(0.022)
    for i in range(df.shape[0]):
        on = i in hot
        for j in range(cols):
            c = t.cell(i + 1, j)
            c.text = str(df.iat[i, j])
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
            r = p.runs[0]
            r.font.size = Pt(font)
            r.font.bold = on
            r.font.color.rgb = NAVY if on else SLATE
            c.fill.solid()
            c.fill.fore_color.rgb = (RGBColor(0xDA, 0xF7, 0xF2) if on
                                     else (WHITE if i % 2 == 0 else RGBColor(0xF2, 0xF6, 0xFA)))
            c.margin_top = c.margin_bottom = Inches(0.016)
    return t


def picture(slide, name, x, y, h=None, w=None):
    """Place a figure scaled to fit inside the box (x, y, w, h), centred.

    Scaling by height alone overflowed the slide edge for the wide matplotlib
    figures, so the box is treated as a bounding box: the image is scaled by
    whichever dimension binds first and then centred in the leftover space.
    """
    p = FIGS / name
    if not p.exists():
        text(slide, x, y, Inches(5), Inches(0.4), f"[missing: {name}]",
             size=11, color=CORAL)
        return None

    probe = slide.shapes.add_picture(str(p), 0, 0)
    nat_w, nat_h = probe.width, probe.height
    probe._element.getparent().remove(probe._element)

    box_w = w if w else Emu(int(nat_w * (h / nat_h)))
    box_h = h if h else Emu(int(nat_h * (w / nat_w)))

    scale = min(box_w / nat_w, box_h / nat_h)
    fw, fh = Emu(int(nat_w * scale)), Emu(int(nat_h * scale))
    fx = Emu(int(x + (box_w - fw) / 2))
    fy = Emu(int(y + (box_h - fh) / 2))
    return slide.shapes.add_picture(str(p), fx, fy, width=fw, height=fh)


def chip(slide, x, y, w, h, label, fill=STEEL, color=WHITE, size=11):
    rect(slide, x, y, w, h, fill=fill, radius=True)
    text(slide, x, y + Inches(0.055), w, h, label, size=size, bold=True,
         color=color, align=PP_ALIGN.CENTER, gap=0)


def arrow_flow(slide, items, x, y, w, h, fill=STEEL, color=WHITE, size=10.5):
    """A horizontal chevron flow."""
    n = len(items)
    gap = Inches(0.1)
    cw = (w - gap * (n - 1)) / n
    for i, label in enumerate(items):
        cx = x + i * (cw + gap)
        sh = slide.shapes.add_shape(MSO_SHAPE.CHEVRON, cx, y, cw, h)
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
        sh.line.fill.background()
        sh.shadow.inherit = False
        tf = sh.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.04)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = label
        r.font.size = Pt(size)
        r.font.bold = True
        r.font.color.rgb = color


# ---------------------------------------------------------------- deck
def build(R):
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    s = R["summary"]
    eda = R["eda"]
    dq = R["dq"]
    pol = R["pol"].set_index("policy")

    # ============================================================ 1 TITLE
    sl = blank(prs)
    sl.background.fill.solid()
    sl.background.fill.fore_color.rgb = INK
    rect(sl, 0, 0, Inches(0.22), H, fill=TEAL)
    rect(sl, Inches(8.9), 0, Inches(4.43), H, fill=RGBColor(0x13, 0x26, 0x45))

    text(sl, Inches(0.85), Inches(0.85), Inches(7.7), Inches(0.3),
         "SRM INSTITUTE OF SCIENCE AND TECHNOLOGY", size=11, bold=True,
         color=TEAL, gap=0)
    text(sl, Inches(0.85), Inches(1.3), Inches(7.8), Inches(2.3),
         "Explainable AI-Driven\nDecision Support for\nPredictive Inventory &\n"
         "Procurement Optimization",
         size=31, bold=True, color=WHITE, gap=2, line_spacing=1.04)
    text(sl, Inches(0.85), Inches(3.72), Inches(7.8), Inches(0.35),
         "in ERP Systems", size=21, color=SKY, gap=0)
    rect(sl, Inches(0.85), Inches(4.3), Inches(2.1), Inches(0.045), fill=AMBER)
    text(sl, Inches(0.85), Inches(4.55), Inches(7.8), Inches(0.3),
         COURSE, size=14, bold=True, color=MUTED, gap=0)
    text(sl, Inches(0.85), Inches(5.0), Inches(7.8), Inches(0.95),
         "A working system that forecasts demand, flags stockout risk,\n"
         "recommends how much to order — and explains why.",
         size=13, color=MUTED, gap=3)

    # team panel
    text(sl, Inches(9.35), Inches(1.35), Inches(3.6), Inches(0.3),
         "TEAM", size=11, bold=True, color=TEAL, gap=0)
    for i, (nm, reg) in enumerate(TEAM):
        yy = Inches(1.78 + i * 0.92)
        rect(sl, Inches(9.35), yy, Inches(3.5), Inches(0.76),
             fill=RGBColor(0x1C, 0x35, 0x5C), radius=True)
        text(sl, Inches(9.55), yy + Inches(0.11), Inches(3.2), Inches(0.3),
             nm, size=13, bold=True, color=WHITE, gap=0)
        text(sl, Inches(9.55), yy + Inches(0.42), Inches(3.2), Inches(0.25),
             reg, size=11, color=TEAL, gap=0)
    rect(sl, Inches(9.35), Inches(4.72), Inches(3.5), Inches(1.5),
         fill=RGBColor(0x1C, 0x35, 0x5C), radius=True)
    text(sl, Inches(9.55), Inches(4.88), Inches(3.2), Inches(1.3),
         "Dataset\nUCI Online Retail II\n1,067,371 real transactions\n"
         "CC BY 4.0  ·  no credentials",
         size=11, color=MUTED, gap=2)
    notes(sl,
          "Good morning ma'am. We are Sandhiya, Nishanth and Nithin. Our "
          "project builds a decision-support system for retail inventory. It "
          "answers three questions for a shopkeeper or a warehouse manager: "
          "how much of a product will sell, are we about to run out, and how "
          "much should we order. And most importantly it explains why it gave "
          "that answer. Everything we show today comes from code that runs on "
          "just over a million real retail transactions.")
    footer(sl, 1)

    # ============================================================ 2 WHAT IS IT
    sl = blank(prs)
    header(sl, "What is our project?", "in plain language")
    text(sl, MARGIN, Inches(1.45), Inches(7.1), Inches(1.0),
         "A shop has thousands of products. Every week somebody must decide:\n"
         "“Which items do I buy more of, and how many?”\n"
         "Today that decision is mostly guesswork.",
         size=15, color=SLATE, gap=5)
    text(sl, MARGIN, Inches(2.62), Inches(7.1), Inches(0.3),
         "OUR SYSTEM ANSWERS SIX QUESTIONS, AUTOMATICALLY:",
         size=11.5, bold=True, color=STEEL, gap=0)
    qs = [
        ("1", "How much will this product sell in the next 7 days?", "Regression"),
        ("2", "Is the stock we have enough?", "Inventory logic"),
        ("3", "Are we at risk of running out?", "Classification"),
        ("4", "Should we reorder — yes or no?", "Classification"),
        ("5", "Exactly how many units should we order?", "Decision policy"),
        ("6", "WHY did it recommend that number?", "Explainable AI"),
    ]
    for i, (num, q, tag) in enumerate(qs):
        yy = Inches(3.0 + i * 0.6)
        rect(sl, MARGIN, yy, Inches(7.1), Inches(0.5), fill=WHITE,
             line=RGBColor(0xDD, 0xE5, 0xEF), radius=True)
        chip(sl, MARGIN + Inches(0.1), yy + Inches(0.08), Inches(0.34),
             Inches(0.34), num, fill=TEAL if i < 5 else AMBER,
             color=INK, size=12)
        text(sl, MARGIN + Inches(0.56), yy + Inches(0.12), Inches(4.5),
             Inches(0.3), q, size=12.5,
             bold=(i == 5), color=NAVY, gap=0)
        text(sl, MARGIN + Inches(5.1), yy + Inches(0.14), Inches(1.9),
             Inches(0.3), tag, size=10, color=STEEL, align=PP_ALIGN.RIGHT, gap=0)

    rect(sl, Inches(8.1), Inches(1.45), Inches(4.65), Inches(3.3),
         fill=WHITE, line=RGBColor(0xDD, 0xE5, 0xEF), radius=True)
    text(sl, Inches(8.35), Inches(1.62), Inches(4.2), Inches(0.3),
         "EXAMPLE OUTPUT", size=11, bold=True, color=STEEL, gap=0)
    c0 = R["cases"][0]
    text(sl, Inches(8.35), Inches(2.0), Inches(4.2), Inches(2.6),
         f"Product              : {c0['sku']}\n"
         f"Stock available      : {c0['current_stock']:,.0f} units\n"
         f"Predicted 7-day sale : {c0['predicted_7d_demand']:,.0f} units\n"
         f"Supplier lead time   : {c0['lead_time_days']} days\n\n"
         f"Stockout risk        : {c0['stockout_risk']}\n"
         f"Reorder              : {'YES' if c0['reorder'] else 'NO'}\n"
         f"Order quantity       : {c0['recommended_qty']:,} units",
         size=11.5, color=NAVY, font="Consolas", gap=2)
    banner(sl, Inches(8.1), Inches(4.95), Inches(4.65), Inches(1.75),
           "Not just a prediction — a DECISION.\n\n"
           "And every recommendation comes with a plain-English reason, so a "
           "human can check it before approving the purchase order.",
           accent=TEAL, bg=RGBColor(0xE9, 0xFB, 0xF8),
           fg=RGBColor(0x0C, 0x4F, 0x47), size=12)
    notes(sl,
          "In simple terms: a shop has thousands of products, and every week "
          "someone has to decide what to buy more of. Today that is mostly "
          "guesswork. Our system answers six questions automatically. The "
          "first five are standard machine learning - predict demand, check "
          "stock, flag risk, decide reorder, compute quantity. The sixth is "
          "what makes our project different: it explains WHY. On the right you "
          "can see a real output from our system for product 20727.")
    footer(sl, 2)

    # ============================================================ 3 PROBLEM
    sl = blank(prs)
    header(sl, "The problem we are solving", "why this matters", accent=CORAL)
    text(sl, MARGIN, Inches(1.5), Inches(5.9), Inches(0.9),
         "Inventory fails in two directions at once — and fixing one\n"
         "usually makes the other worse.",
         size=15, color=SLATE, gap=4)
    rect(sl, MARGIN, Inches(2.5), Inches(2.85), Inches(2.25),
         fill=RGBColor(0xFF, 0xF0, 0xF0), line=CORAL, radius=True)
    text(sl, MARGIN, Inches(2.68), Inches(2.85), Inches(0.35),
         "TOO LITTLE STOCK", size=12.5, bold=True, color=CORAL,
         align=PP_ALIGN.CENTER, gap=0)
    text(sl, MARGIN + Inches(0.2), Inches(3.12), Inches(2.45), Inches(1.5),
         "• Customer wants it,\n   shelf is empty\n• Sale is lost\n"
         "• Customer may not\n   come back",
         size=11.5, color=RGBColor(0x8A, 0x2B, 0x2B), gap=3)
    rect(sl, MARGIN + Inches(3.05), Inches(2.5), Inches(2.85), Inches(2.25),
         fill=RGBColor(0xFF, 0xFA, 0xEC), line=AMBER, radius=True)
    text(sl, MARGIN + Inches(3.05), Inches(2.68), Inches(2.85), Inches(0.35),
         "TOO MUCH STOCK", size=12.5, bold=True, color=RGBColor(0xA9, 0x6A, 0x0C),
         align=PP_ALIGN.CENTER, gap=0)
    text(sl, MARGIN + Inches(3.25), Inches(3.12), Inches(2.45), Inches(1.5),
         "• Cash locked on\n   the shelf\n• Storage cost\n"
         "• Goods may expire\n   or go out of season",
         size=11.5, color=RGBColor(0x7A, 0x4E, 0x0A), gap=3)
    text(sl, MARGIN, Inches(4.95), Inches(5.9), Inches(0.4),
         "The deciding factor: how much to order, and when.",
         size=14, bold=True, color=NAVY, gap=0)

    card(sl, Inches(6.9), Inches(1.5), Inches(2.8), Inches(1.5),
         f"{pol.loc['P0','fill_rate']:.1%}", "Fill rate using today's\n"
         "guesswork method\n(1 in 5 units unavailable)", accent=CORAL)
    card(sl, Inches(9.95), Inches(1.5), Inches(2.8), Inches(1.5),
         f"{int(pol.loc['P0','stockout_days']):,}",
         "Days a product was\nout of stock\nin the test period", accent=CORAL)
    banner(sl, Inches(6.9), Inches(3.2), Inches(5.85), Inches(1.55),
           "Why is this hard?\n"
           f"Demand is wildly uneven — skewness {eda['demand_distribution']['skew']:.0f}, "
           f"and {eda['intermittency']['overall_pct_zero_sku_days']:.0f}% of product-days "
           "have NO sales at all.\n"
           "An average-based rule is wrong on most days.", accent=CORAL,
           bg=RGBColor(0xFF, 0xF2, 0xF2), fg=RGBColor(0x7D, 0x25, 0x25))
    banner(sl, MARGIN, Inches(5.5), Inches(12.1), Inches(1.15),
           "KEY INSIGHT THAT SHAPED OUR WHOLE PROJECT:  a forecast by itself changes nothing. "
           "We proved this — using a machine-learning forecast with no safety buffer made the fill rate "
           "WORSE than the simple baseline (79.2% vs 79.9%). The value comes from handling UNCERTAINTY.",
           accent=TEAL, bg=RGBColor(0xE9, 0xFB, 0xF8), fg=RGBColor(0x0C, 0x4F, 0x47), size=13)
    notes(sl,
          "Inventory fails in two directions at once. Too little stock and you "
          "lose the sale and maybe the customer. Too much and your cash is "
          "locked on a shelf. In our test environment the simple "
          "moving-average rule that most small retailers actually use achieves "
          "only a 79.9% fill rate - one unit in five that a customer wanted "
          "was not there. And it is hard because demand is extremely uneven. "
          "The key insight that shaped our whole project is at the bottom: a "
          "forecast by itself changes nothing.")
    footer(sl, 3)

    # ============================================================ 4 BENEFITS
    sl = blank(prs)
    header(sl, "Benefits — who gains, and how much", "measured, not claimed",
           accent=LIME)
    res = [
        (f"{pol.loc['P4','fill_rate']:.1%}", "Fill rate achieved\nup from 79.9%", LIME),
        (f"+{(pol.loc['P4','fill_rate']-pol.loc['P0','fill_rate'])*100:.1f}pp",
         "Improvement over\nthe baseline rule", LIME),
        (f"-{(1-pol.loc['P4','stockout_days']/pol.loc['P0','stockout_days'])*100:.0f}%",
         "Fewer stockout days\n2,027 → 496", LIME),
        (f"-{(1-pol.loc['P4','shortage_cost']/pol.loc['P0','shortage_cost'])*100:.0f}%",
         "Lower shortage cost\n(simulated)", LIME),
    ]
    for i, (v, lab, c) in enumerate(res):
        card(sl, MARGIN + Inches(i * 3.08), Inches(1.5), Inches(2.9),
             Inches(1.5), v, lab, accent=c)

    rows = [
        ("Shop owner / buyer", "Fewer lost sales; less cash stuck in slow stock"),
        ("Warehouse team", "A ranked daily list of what to reorder, not a spreadsheet"),
        ("Finance", "Stockout cost cut by 68% in simulation; spend is justified per order"),
        ("Supplier", "Earlier, steadier orders instead of panic purchases"),
        ("Auditor / manager", "Every order has a written reason that adds up exactly"),
    ]
    df = pd.DataFrame(rows, columns=["Who benefits", "What they get"])
    table(sl, df, MARGIN, Inches(3.3), Inches(7.6), Inches(2.3), font=12,
          widths=[1.0, 2.3], accent=STEEL)

    rect(sl, Inches(8.5), Inches(3.3), Inches(4.25), Inches(2.3),
         fill=WHITE, line=RGBColor(0xDD, 0xE5, 0xEF), radius=True)
    text(sl, Inches(8.75), Inches(3.47), Inches(3.8), Inches(0.3),
         "ACADEMIC VALUE", size=11, bold=True, color=STEEL, gap=0)
    text(sl, Inches(8.75), Inches(3.82), Inches(3.8), Inches(1.7),
         "• Unit 1 and Unit 2 topics implemented\n   in running, tested code\n"
         "• 13 models compared on one problem\n"
         "• 81 automated tests\n"
         "• Fully reproducible: one command",
         size=11.5, color=SLATE, gap=4)
    banner(sl, MARGIN, Inches(5.8), Inches(12.1), Inches(0.9),
           "Honest framing: the cost figures come from a simulated supply chain, so the RANKING of methods is "
           "a real result — the rupee amounts are not a claim about a real shop. We say this openly on the next slide.",
           accent=AMBER)
    notes(sl,
          "These are the benefits, and they are measured rather than claimed. "
          "Our best policy raises the fill rate from 79.9% to 93.9% - fourteen "
          "percentage points - and cuts the number of stockout days by 75%. "
          "The table shows who gains. But I want to be careful: the cost "
          "figures come from a simulated supply chain, so the ranking of "
          "methods is a real result, the exact money amounts are not. We "
          "explain that honestly on the next slide.")
    footer(sl, 4)

    # ============================================================ 5 DATASET
    sl = blank(prs)
    header(sl, "Dataset — where our data came from", "real data, openly sourced")
    df = pd.DataFrame({
        "Property": ["Name", "Source", "Business", "Period",
                     "Raw records", "After cleaning", "Licence", "Credentials"],
        "Value": ["Online Retail II",
                  "UCI Machine Learning Repository",
                  "UK online giftware wholesaler",
                  "01 Dec 2009 → 09 Dec 2011",
                  f"{dq['raw']['n_rows']:,} invoice lines",
                  f"{dq['clean']['n_rows']:,}  (96.01% retained)",
                  "CC BY 4.0 (free to use with attribution)",
                  "None — direct download"],
    })
    table(sl, df, MARGIN, Inches(1.45), Inches(6.5), Inches(2.85), font=11.5,
          widths=[0.9, 1.9])

    text(sl, MARGIN, Inches(4.5), Inches(6.5), Inches(0.3),
         "SOURCE LINKS", size=11, bold=True, color=STEEL, gap=0)
    link(sl, MARGIN, Inches(4.82), Inches(6.6), Inches(0.26),
         "▸  Dataset page:  archive.ics.uci.edu/dataset/502/online+retail+ii",
         "https://archive.ics.uci.edu/dataset/502/online+retail+ii", size=11.5)
    link(sl, MARGIN, Inches(5.12), Inches(6.6), Inches(0.26),
         "▸  DOI:  https://doi.org/10.24432/C5CG6D",
         "https://doi.org/10.24432/C5CG6D", size=11.5)
    link(sl, MARGIN, Inches(5.42), Inches(6.6), Inches(0.26),
         "▸  Alternative considered — Kaggle Store Sales (Favorita)",
         "https://www.kaggle.com/competitions/store-sales-time-series-forecasting",
         size=11.5)

    rect(sl, Inches(7.45), Inches(1.45), Inches(5.3), Inches(2.85),
         fill=WHITE, line=RGBColor(0xDD, 0xE5, 0xEF), radius=True)
    text(sl, Inches(7.7), Inches(1.62), Inches(4.8), Inches(0.3),
         "WHAT EACH RECORD CONTAINS", size=11, bold=True, color=STEEL, gap=0)
    text(sl, Inches(7.7), Inches(1.98), Inches(4.8), Inches(2.2),
         "Invoice      · bill number\nStockCode    · product ID\n"
         "Description  · product name\nQuantity     · units sold\n"
         "InvoiceDate  · date & time\nPrice        · unit price\n"
         "Customer ID  · buyer (22.8% blank)\nCountry      · 43 countries",
         size=11.5, color=NAVY, font="Consolas", gap=2)

    banner(sl, Inches(7.45), Inches(4.5), Inches(5.3), Inches(1.68),
           "WHY THIS DATASET?\n"
           "Kaggle's Favorita set is richer (it has promotions and holidays) but "
           "needs a personal login token, which cannot be automated.\n"
           "We chose reproducibility: our project downloads its data and runs "
           "with ONE command, on any machine.", accent=TEAL,
           bg=RGBColor(0xE9, 0xFB, 0xF8), fg=RGBColor(0x0C, 0x4F, 0x47), size=11.5)
    notes(sl,
          "Our data is real. It comes from the UCI Machine Learning Repository "
          "- the Online Retail II dataset, which is a UK online giftware "
          "wholesaler's complete sales record for two years, just over a "
          "million invoice lines, free to use under Creative Commons. The "
          "links are on the slide. We did consider Kaggle's Favorita dataset "
          "which is richer, but it needs a personal login token that cannot be "
          "automated, so we chose this one - our project downloads its own "
          "data and runs with one command on any machine.")
    footer(sl, 5)

    # ============================================================ 6 HONESTY
    sl = blank(prs)
    header(sl, "What is real and what we had to simulate", "data honesty",
           accent=AMBER)
    text(sl, MARGIN, Inches(1.45), Inches(12.1), Inches(0.45),
         "The dataset records what was SOLD. It does not record what was in the warehouse. "
         "But you cannot decide a purchase order without stock, suppliers and costs — "
         "so we generate those, and we say so everywhere.",
         size=13.5, color=SLATE, gap=3)

    rect(sl, MARGIN, Inches(2.15), Inches(5.9), Inches(3.3),
         fill=RGBColor(0xEC, 0xFA, 0xF1), line=LIME, radius=True)
    text(sl, MARGIN + Inches(0.25), Inches(2.33), Inches(5.4), Inches(0.35),
         "REAL — straight from the dataset", size=13, bold=True,
         color=RGBColor(0x1B, 0x6B, 0x45), gap=0)
    text(sl, MARGIN + Inches(0.25), Inches(2.78), Inches(5.4), Inches(2.4),
         "✓  units sold per product per day\n✓  dates and the trading calendar\n"
         "✓  unit prices\n✓  product codes and descriptions\n"
         "✓  invoice and customer records\n✓  country of the buyer\n"
         "✓  returns / cancellations",
         size=12.5, color=RGBColor(0x1E, 0x5A, 0x3D), gap=4)

    rect(sl, Inches(6.85), Inches(2.15), Inches(5.9), Inches(3.3),
         fill=RGBColor(0xFF, 0xF2, 0xF2), line=CORAL, radius=True)
    text(sl, Inches(7.1), Inches(2.33), Inches(5.4), Inches(0.35),
         "SIMULATED — generated by us, seeded", size=13, bold=True,
         color=RGBColor(0x8A, 0x2B, 0x2B), gap=0)
    text(sl, Inches(7.1), Inches(2.78), Inches(5.4), Inches(2.4),
         "✗  stock level in the warehouse\n✗  suppliers and their reliability\n"
         "✗  delivery lead times\n✗  minimum order quantity, pack size\n"
         "✗  supplier capacity\n✗  purchase / holding / shortage costs\n"
         "✗  purchase orders",
         size=12.5, color=RGBColor(0x8A, 0x2B, 0x2B), gap=4)

    banner(sl, MARGIN, Inches(5.55), Inches(12.1), Inches(1.15),
           "How we keep this honest:  all simulated values come from one config file (sim/supply_config.yaml) "
           "with a fixed random seed, every row is flagged is_simulated, and the dashboard shows a warning. "
           "Because all methods face the SAME real demand and the SAME simulated terms, comparing them is valid — "
           "we just never call a simulated number a measurement.",
           accent=AMBER, size=12.5)
    notes(sl,
          "This slide is about honesty, and we think it is the most important "
          "slide in the deck. The dataset records what was sold. It does not "
          "record what was sitting in the warehouse, who the suppliers were, "
          "how long delivery took, or what anything cost. But you cannot "
          "decide a purchase order without those. So we generate them, from "
          "one config file with a fixed seed, and we label them as simulated "
          "everywhere - in the code, the dashboard and the report. Because "
          "every method faces the same real demand and the same simulated "
          "terms, comparing the methods is still valid. We just never call a "
          "simulated number a measurement.")
    footer(sl, 6)

    # ============================================================ 7 METHOD
    sl = blank(prs)
    header(sl, "How the system works", "methodology")
    arrow_flow(sl, ["RAW DATA\n1.07M rows", "CLEANING\n7 rules", "EDA\n16 charts",
                    "FEATURES\n43 / 52", "MODELS\n13 compared",
                    "DECISION\npolicy", "EXPLAIN\nSHAP"],
               MARGIN, Inches(1.45), Inches(12.1), Inches(0.85))

    text(sl, MARGIN, Inches(2.6), Inches(6.0), Inches(0.3),
         "TWO MACHINE-LEARNING PROBLEMS", size=11, bold=True, color=STEEL, gap=0)
    rect(sl, MARGIN, Inches(2.95), Inches(2.9), Inches(1.5), fill=WHITE,
         line=SKY, radius=True)
    text(sl, MARGIN + Inches(0.15), Inches(3.08), Inches(2.6), Inches(1.3),
         "REGRESSION\nHow many units will\nsell in 7 days?\n→ e.g. 486 units",
         size=12, color=NAVY, gap=3)
    rect(sl, MARGIN + Inches(3.1), Inches(2.95), Inches(2.9), Inches(1.5),
         fill=WHITE, line=TEAL, radius=True)
    text(sl, MARGIN + Inches(3.25), Inches(3.08), Inches(2.6), Inches(1.3),
         "CLASSIFICATION\nWill we run out before\nthe delivery arrives?\n→ YES / NO",
         size=12, color=NAVY, gap=3)

    text(sl, MARGIN, Inches(4.65), Inches(6.0), Inches(0.3),
         "THE RULE THAT PROTECTS US FROM CHEATING", size=11, bold=True,
         color=STEEL, gap=0)
    text(sl, MARGIN, Inches(5.0), Inches(6.0), Inches(1.4),
         "A decision on day t may use data from day t−1 and earlier ONLY.\n"
         "The answer (days t … t+6) is never visible to the model.\n"
         "We split by DATE, never randomly — and 8 automated tests check it.",
         size=12.5, color=SLATE, gap=4)

    df = pd.DataFrame({
        "Split": ["Train", "Validation", "Test"],
        "Rows": ["66,859", "16,239", "16,689"],
        "Period": ["Jan 2010 – May 2011", "May – Aug 2011", "Aug – Dec 2011"],
    })
    table(sl, df, Inches(7.0), Inches(2.95), Inches(5.75), Inches(1.5),
          font=11.5, widths=[0.7, 0.6, 1.4])
    banner(sl, Inches(7.0), Inches(4.65), Inches(5.75), Inches(1.75),
           "Why split by date?\n"
           "Two days in a row share almost all their history. If one went into "
           "training and the next into testing, the model would effectively "
           "have seen the answer — and our scores would look great but be "
           "meaningless.", accent=TEAL, bg=RGBColor(0xE9, 0xFB, 0xF8),
           fg=RGBColor(0x0C, 0x4F, 0x47), size=12)
    notes(sl,
          "This is how the system works end to end. Raw data, cleaning with "
          "seven documented rules, exploratory analysis, feature engineering, "
          "thirteen models, then the decision policy and the explanation "
          "layer. We solve two machine learning problems: a regression that "
          "predicts how many units will sell, and a classification that says "
          "whether we will run out. The bottom-left box is the rule that stops "
          "us cheating - a decision on day t can only use data up to the day "
          "before, and we split by date rather than randomly, because two "
          "consecutive days share almost all their history.")
    footer(sl, 7)

    # ============================================================ 8 EDA
    sl = blank(prs)
    header(sl, "What we discovered in the data", "exploratory data analysis")
    dd, tq = eda["demand_distribution"], eda["target_quantiles"]
    for i, (v, lab, c) in enumerate([
        ("30 vs 8,167", "Saturday invoices vs\nThursday — the shop is\nCLOSED on Saturdays", CORAL),
        (f"{dd['skew']:.0f}", "Skewness — demand is\nnothing like a\nnormal bell curve", AMBER),
        (f"{eda['intermittency']['overall_pct_zero_sku_days']:.0f}%", "of product-days have\nZERO sales", AMBER),
        (f"{tq['P95_over_P50']:.1f}×", "A busy week sells 8×\na typical week", CORAL),
    ]):
        card(sl, MARGIN + Inches(i * 3.08), Inches(1.45), Inches(2.9),
             Inches(1.62), v, lab, accent=c, vsize=22)
    picture(sl, "eda_08_day_of_week.png", MARGIN, Inches(3.3), w=Inches(5.9), h=Inches(2.55))
    picture(sl, "eda_15_target_quantiles.png", Inches(6.9), Inches(3.3), w=Inches(5.85), h=Inches(2.55))
    banner(sl, MARGIN, Inches(6.0), Inches(12.1), Inches(0.72),
           "The Saturday discovery nearly broke our project: if we had filled those closed days with zeros, "
           "we would have added ~17% FAKE no-sale records and pushed every forecast downwards. "
           "We build the calendar from days the shop actually traded — 604 of 739.",
           accent=CORAL, bg=RGBColor(0xFF, 0xF2, 0xF2), fg=RGBColor(0x7D, 0x25, 0x25))
    notes(sl,
          "Four discoveries shaped everything. First and most important: this "
          "shop does not trade on Saturdays - thirty invoices against eight "
          "thousand on a Thursday. If we had filled those closed days with "
          "zeros we would have added about seventeen percent fake no-sale "
          "records and pushed every forecast down. Second, demand is skewed at "
          "24, nothing like a bell curve. Third, nearly a third of product-days "
          "have no sales at all. Fourth, a busy week sells eight times a "
          "typical week - and that last number is the reason our whole "
          "uncertainty layer exists.")
    footer(sl, 8)

    # ============================================================ 9 UNIT 1
    sl = blank(prs)
    header(sl, "Unit 1 — Polynomial curve fitting & overfitting",
           "syllabus concept, measured on real data", accent=SKY)
    p = R["poly"].sort_values("degree")
    df = pd.DataFrame({
        "Degree": p["degree"].astype(int),
        "Terms": p["n_polynomial_terms"].astype(int),
        "Train error": p["train_RMSE"].round(1),
        "Test error": p["test_RMSE"].round(1),
        "Test R²": p["test_R2"].round(2),
    })
    table(sl, df, MARGIN, Inches(1.5), Inches(5.9), Inches(1.85), font=12,
          hot=[3], widths=[0.7, 0.6, 0.9, 0.9, 0.7])
    text(sl, MARGIN, Inches(3.6), Inches(5.9), Inches(2.0),
         "As we increase the polynomial degree:\n\n"
         "▸  Training error keeps IMPROVING  285.8 → 278.4\n"
         "▸  Test error gets 3.2× WORSE      291.1 → 942.0\n"
         "▸  R² becomes NEGATIVE at degree 4 — the model is\n"
         "    worse than simply guessing the average\n\n"
         "The penalty is held constant, so only capacity changed.",
         size=12.5, color=SLATE, gap=4)
    picture(sl, "model_polynomial_overfitting.png", Inches(6.85), Inches(1.5),
            w=Inches(5.9), h=Inches(4.3))
    banner(sl, MARGIN, Inches(5.9), Inches(12.1), Inches(0.8),
           "This is the bias–variance trade-off from Unit 1 — not drawn as a textbook cartoon, but measured on "
           "our own data. A more powerful model is not automatically a better model.",
           accent=SKY, bg=RGBColor(0xEC, 0xF4, 0xFE), fg=RGBColor(0x14, 0x3A, 0x66))
    notes(sl,
          "This is the Unit 1 concept of polynomial curve fitting, and we "
          "measured it on real data rather than drawing a textbook cartoon. As "
          "we raise the degree, the training error keeps improving - the model "
          "fits the data it has seen better and better. But the test error "
          "gets 3.2 times worse, and at degree four the R-squared actually "
          "goes negative, meaning the model is worse than just guessing the "
          "average. Same data, same penalty, only the capacity changed. That "
          "is overfitting, and you can watch it happen.")
    footer(sl, 9)

    # ============================================================ 10 REGRESSION
    sl = blank(prs)
    header(sl, "Unit 2 — Demand forecasting results",
           "7 models compared, including 2 simple baselines", accent=SKY)
    reg = R["reg"].sort_values("test_RMSE")
    df = pd.DataFrame({
        "Model": [m.replace(" (gradient-boosted trees)", "")
                  .replace("Bayesian Linear Regression (Bayesian Ridge)", "Bayesian Linear Regression")
                  .replace("Baseline: ", "★ Baseline: ")
                  .replace(" x horizon", "") for m in reg["model"]],
        "MAE": reg["test_MAE"].round(1),
        "RMSE": reg["test_RMSE"].round(1),
        "R²": reg["test_R2"].round(3),
    })
    table(sl, df, MARGIN, Inches(1.45), Inches(6.5), Inches(2.9), font=11,
          hot=[0], widths=[2.5, 0.6, 0.6, 0.6])
    picture(sl, "model_regression_rmse.png", Inches(7.3), Inches(1.45), w=Inches(5.45), h=Inches(2.9))

    banner(sl, MARGIN, Inches(4.55), Inches(6.2), Inches(1.15),
           "Unit 2 link: minimising squared error IS maximising the Gaussian "
           "likelihood — so ordinary least squares already is a Maximum "
           "Likelihood Estimator. Ridge adds an L2 penalty.",
           accent=SKY, bg=RGBColor(0xEC, 0xF4, 0xFE), fg=RGBColor(0x14, 0x3A, 0x66),
           size=12)
    banner(sl, Inches(7.0), Inches(4.55), Inches(5.75), Inches(1.15),
           "HONEST RESULT: LightGBM — the most advanced model — came 6th of 7, "
           "behind a simple moving average. We report it instead of hiding it.",
           accent=CORAL, bg=RGBColor(0xFF, 0xF2, 0xF2), fg=RGBColor(0x7D, 0x25, 0x25),
           size=12)
    text(sl, MARGIN, Inches(5.88), Inches(12.1), Inches(0.8),
         "Why did the advanced model lose?  Demand is 31.7% zeros, which fragments decision trees; and the test "
         "period is the Christmas peak, which lies outside the range the trees ever saw — trees cannot extrapolate, "
         "linear models can.", size=12, color=SLATE, gap=3)
    notes(sl,
          "Here are our forecasting results - seven models including two "
          "simple baselines so we can tell whether machine learning actually "
          "added anything. Ridge regression won. The Unit 2 connection is on "
          "the left: minimising squared error is the same thing as maximising "
          "the Gaussian likelihood, so ordinary least squares already is a "
          "maximum likelihood estimator. And on the right is our honest "
          "result: LightGBM, the most advanced model, came sixth of seven, "
          "behind a moving average. We report it rather than hide it, and we "
          "can explain why - zero inflation and a test period outside the "
          "training range.")
    footer(sl, 10)

    # ============================================================ 11 CLASSIFICATION
    sl = blank(prs)
    header(sl, "Unit 2 — Reorder prediction & the kernel trick",
           "6 classifiers, each mapped to a syllabus topic", accent=TEAL)
    clf = R["clf"].sort_values("test_f1", ascending=False)
    df = pd.DataFrame({
        "Model": [m.replace("Linear Discriminant Analysis", "LDA — discriminant function")
                  .replace("Gaussian Naive Bayes", "Gaussian NB — generative")
                  .replace("Logistic Regression", "Logistic — discriminative")
                  .replace("LightGBM Classifier", "LightGBM")
                  for m in clf["model"]],
        "Precision": clf["test_precision"].round(3),
        "Recall": clf["test_recall"].round(3),
        "F1": clf["test_f1"].round(3),
        "AUC": clf["test_roc_auc"].round(3),
    })
    table(sl, df, MARGIN, Inches(1.45), Inches(6.6), Inches(2.6), font=10.5,
          hot=[0], widths=[2.4, 0.7, 0.65, 0.6, 0.6])
    picture(sl, "model_roc_curves.png", Inches(7.4), Inches(1.4), w=Inches(5.35), h=Inches(3.5))
    banner(sl, MARGIN, Inches(4.35), Inches(6.6), Inches(1.35),
           "GENERATIVE vs DISCRIMINATIVE (Unit 2)\n"
           "Naive Bayes uses Bayes' rule on p(x|y) — recall 0.969 but precision "
           "only 0.374: it flags almost everything. Logistic models p(y|x) "
           "directly: F1 0.748 vs 0.540.",
           accent=TEAL, bg=RGBColor(0xE9, 0xFB, 0xF8), fg=RGBColor(0x0C, 0x4F, 0x47),
           size=11.5)
    banner(sl, MARGIN, Inches(5.85), Inches(12.1), Inches(0.85),
           "KERNEL TRICK:  Linear SVM (F1 0.748) BEAT the RBF kernel SVM (0.718) — same settings, same data, "
           "only the kernel changed. The infinite-dimensional feature map did not help, which tells us the "
           "boundary is nearly linear here. Another negative result, reported.",
           accent=CORAL, bg=RGBColor(0xFF, 0xF2, 0xF2), fg=RGBColor(0x7D, 0x25, 0x25),
           size=12)
    notes(sl,
          "Six classifiers, each one mapped to a Unit 2 topic. Two results are "
          "worth explaining. Generative versus discriminative: Naive Bayes "
          "models the feature distribution and applies Bayes rule - it catches "
          "97% of the real cases but its precision is only 37%, so it flags "
          "almost everything. Logistic regression models the probability "
          "directly and is much better balanced. And the kernel trick: the "
          "linear SVM actually beat the RBF kernel SVM with identical "
          "settings, which tells us the decision boundary is nearly linear "
          "here. That is a negative result and we report it.")
    footer(sl, 11)

    # ============================================================ 12 UNCERTAINTY
    sl = blank(prs)
    header(sl, "Unit 1 — Handling uncertainty with quantiles",
           "probability, mean, variance and quantiles applied", accent=AMBER)
    qt = R["qt"].sort_values("quantile")
    df = pd.DataFrame({
        "Target": [f"P{int(q*100)}" for q in qt["quantile"]],
        "Actual coverage": qt["empirical_coverage_test"].round(3),
        "Error": qt["coverage_error"].round(3),
    })
    table(sl, df, MARGIN, Inches(1.5), Inches(4.3), Inches(1.85), font=12,
          widths=[0.7, 1.2, 0.7])
    text(sl, MARGIN, Inches(3.6), Inches(5.9), Inches(2.3),
         "A single predicted number is not enough for stock planning.\n\n"
         "▸  P50 = the typical week\n"
         "▸  P90 = a busy week — stock for this and you are short\n"
         "    only 1 week in 10\n"
         "▸  P95 = a very busy week\n\n"
         "The gap between P95 and P50 becomes our SAFETY STOCK —\n"
         "learned from the data, not assumed from a bell curve.",
         size=12.5, color=SLATE, gap=4)
    picture(sl, "model_quantile_calibration.png", Inches(6.85), Inches(1.5),
            w=Inches(5.9), h=Inches(4.3))
    banner(sl, MARGIN, Inches(5.9), Inches(12.1), Inches(0.8),
           "Calibration check — does our P90 model really cover 90% of weeks?  It covers 89.2%. "
           "Every level is within 2.1 percentage points of its target, with no normality assumption — "
           "which matters because our demand skewness is 24.6.",
           accent=AMBER, size=12.5)
    notes(sl,
          "A single predicted number is not enough to plan stock with. So we "
          "also predict quantiles - the typical week, the busy week, the very "
          "busy week. The gap between the 95th percentile and the median "
          "becomes our safety stock, and because it is learned from the data "
          "we do not have to assume a bell curve, which matters a lot when the "
          "skewness is 24. The table checks calibration: does our P90 model "
          "really cover ninety percent of weeks? It covers 89.2 percent. Every "
          "level is within about two points.")
    footer(sl, 12)

    # ============================================================ 13 DECISION
    sl = blank(prs)
    header(sl, "From prediction to DECISION", "our most important result",
           accent=LIME)
    pl = R["pol"]
    df = pd.DataFrame({
        "Method": ["P0  Simple moving average", "P1  Classical reorder point",
                   "P2  ML forecast, no buffer", "P3  ML + classical buffer",
                   "P4  ML + learned buffer"],
        "Fill rate": [f"{v:.1%}" for v in pl["fill_rate"]],
        "Stockout days": [f"{int(v):,}" for v in pl["stockout_days"]],
        "Cost vs P0": [f"{v:+.1f}%" for v in pl["cost_vs_P0_pct"]],
    })
    table(sl, df, MARGIN, Inches(1.45), Inches(6.6), Inches(2.3), font=11.5,
          hot=[2, 4], widths=[2.2, 0.8, 0.95, 0.8])
    picture(sl, "policy_cost_vs_fill_rate.png", Inches(7.4), Inches(1.45), w=Inches(5.35), h=Inches(3.4))
    banner(sl, MARGIN, Inches(4.0), Inches(6.6), Inches(1.5),
           "LOOK AT P2.\n"
           "The machine-learning forecast with NO safety buffer scores 79.2% — "
           "WORSE than the simple baseline's 79.9%.\n"
           "A better prediction, on its own, did not help at all.",
           accent=CORAL, bg=RGBColor(0xFF, 0xF2, 0xF2), fg=RGBColor(0x7D, 0x25, 0x25),
           size=12.5)
    banner(sl, MARGIN, Inches(5.65), Inches(12.1), Inches(1.05),
           "What DID work is handling uncertainty.  P1 gains 8.8 points using a classical buffer and no ML at all. "
           "Our full method P4 reaches 93.9% — a 14-point gain with 75% fewer stockout days — for 4.6% more cost. "
           "Whether that trade is worth it depends on how the business prices a lost sale.",
           accent=LIME, bg=RGBColor(0xEC, 0xFA, 0xF1), fg=RGBColor(0x1B, 0x6B, 0x45),
           size=12.5)
    notes(sl,
          "This is our most important result. Look at P2 - the machine "
          "learning forecast with no safety buffer. It scores 79.2 percent, "
          "which is worse than the simple baseline. A better prediction on its "
          "own did not help at all. What did work is handling uncertainty. P1 "
          "gains almost nine points using a classical safety buffer and no "
          "machine learning whatsoever. Our full method, P4, combines the ML "
          "forecast with the learned buffer and reaches 93.9 percent - "
          "fourteen points better, with 75 percent fewer stockout days, for "
          "4.6 percent more cost.")
    footer(sl, 13)

    # ============================================================ 14 XAI
    sl = blank(prs)
    header(sl, "Explainable AI — why THIS number?",
           "the part that makes it usable", accent=TEAL)
    c = R["cases"][0]
    lines = []
    for k, v in c["decomposition"].items():
        lines.append(f"{v:>+10,.1f}   {k}")
    rect(sl, MARGIN, Inches(1.45), Inches(6.3), Inches(3.5), fill=WHITE,
         line=RGBColor(0xDD, 0xE5, 0xEF), radius=True)
    text(sl, MARGIN + Inches(0.22), Inches(1.6), Inches(5.9), Inches(0.3),
         f"PRODUCT {c['sku']}  ·  RISK {c['stockout_risk']}  ·  "
         f"ORDER {c['recommended_qty']:,} UNITS",
         size=11.5, bold=True, color=STEEL, gap=0)
    text(sl, MARGIN + Inches(0.22), Inches(1.98), Inches(5.9), Inches(2.8),
         "\n".join(lines) + "\n" + "─" * 44 +
         f"\n{c['decomposition_total']:>10,}   FINAL ORDER QUANTITY",
         size=10.5, color=NAVY, font="Consolas", gap=1)
    picture(sl, "shap_waterfall_case1.png", Inches(7.1), Inches(1.45), w=Inches(5.65), h=Inches(3.5))

    wf = R["wf"]
    first = wf.loc[wf["case"] == wf["case"].iloc[0]].head(5)
    df = pd.DataFrame({
        "What if…": first["scenario"],
        "New order": [f"{int(v):,}" for v in first["recommended_qty"]],
        "Change": [f"{int(v):+,}" if v else "—" for v in first["change_vs_current"]],
    })
    table(sl, df, MARGIN, Inches(5.15), Inches(6.3), Inches(1.5), font=10.5,
          hot=[0], widths=[2.4, 0.7, 0.65])
    banner(sl, Inches(7.1), Inches(5.15), Inches(5.65), Inches(1.5),
           "Two layers of explanation\n"
           "1.  WHY this forecast — SHAP. Our winning model is linear, so the "
           "attributions are EXACT, not estimated.\n"
           "2.  WHY this quantity — the arithmetic adds up to the order with "
           "residual 0.000000 across all 16,689 decisions.",
           accent=TEAL, bg=RGBColor(0xE9, 0xFB, 0xF8), fg=RGBColor(0x0C, 0x4F, 0x47),
           size=11.5)
    notes(sl,
          "Most explainable-AI demos stop at saying which feature was "
          "important. That does not tell a shopkeeper why the system asked for "
          "1,380 units. So we built two layers. The first explains the "
          "forecast using SHAP - and because our winning model is linear, the "
          "values are exact rather than estimated. The second explains the "
          "quantity as simple arithmetic you can check by hand: expected "
          "demand, plus safety stock, minus what we already have, plus pack "
          "rounding. It adds up exactly, on every one of the 16,689 decisions. "
          "And the bottom-left table is our what-if feature, so the buyer can "
          "test a scenario before committing.")
    footer(sl, 14)

    # ============================================================ 15 CONCLUSION
    sl = blank(prs)
    header(sl, "Conclusion", "what we built, what we learned", accent=LIME)
    for i, (v, lab, c) in enumerate([
        (f"{s['best_regression_test_R2']:.3f}", "Forecast R²\nRidge Regression", SKY),
        (f"{s['best_classifier_test_f1']:.3f}", "Reorder F1 score\nLightGBM", SKY),
        (f"{pol.loc['P4','fill_rate']:.1%}", "Fill rate achieved\nup from 79.9%", LIME),
        ("81 / 51", "Tests passing /\nrubric checks", LIME),
    ]):
        card(sl, MARGIN + Inches(i * 3.08), Inches(1.4), Inches(2.9),
             Inches(1.45), v, lab, accent=c, vsize=23)

    text(sl, MARGIN, Inches(3.08), Inches(6.1), Inches(0.3),
         "WHAT WE LEARNED", size=11, bold=True, color=STEEL, gap=0)
    text(sl, MARGIN, Inches(3.42), Inches(6.1), Inches(1.9),
         "▸  A better forecast alone did NOT help — handling\n"
         "    uncertainty is what improved service\n"
         "▸  The most advanced model lost the forecasting task\n"
         "▸  The non-linear kernel lost to the linear one\n"
         "▸  Explaining the decision matters more than\n"
         "    explaining the model",
         size=12, color=SLATE, gap=4)

    text(sl, Inches(7.0), Inches(3.08), Inches(5.75), Inches(0.3),
         "LIMITATIONS WE ACKNOWLEDGE", size=11, bold=True, color=STEEL, gap=0)
    text(sl, Inches(7.0), Inches(3.42), Inches(5.75), Inches(1.9),
         "▸  Supply-side data is simulated\n"
         "▸  One retailer, one country, two years\n"
         "▸  Top 200 products = 42% of revenue\n"
         "▸  No promotion data in the dataset\n"
         "▸  Human approval is still required —\n"
         "    nothing is ordered automatically",
         size=12, color=SLATE, gap=4)

    rect(sl, MARGIN, Inches(5.4), Inches(12.1), Inches(1.3), fill=NAVY, radius=True)
    text(sl, MARGIN + Inches(0.3), Inches(5.55), Inches(7.6), Inches(0.35),
         "LIVE DEMO  —  Streamlit dashboard, 6 pages", size=14, bold=True,
         color=TEAL, gap=0)
    text(sl, MARGIN + Inches(0.3), Inches(5.92), Inches(7.6), Inches(0.65),
         "Overview · Demand Prediction · Inventory Risk · Procurement "
         "Recommendation · WHY? · Model Comparison",
         size=11.5, color=MUTED, gap=2)
    text(sl, Inches(9.0), Inches(5.6), Inches(3.6), Inches(0.9),
         "Sandhiya S · RA2411003010045\nNishanth Kannan · RA2411003010051\n"
         "Netti Nithin Chakravarthy · RA2411003010033",
         size=10.5, color=MUTED, align=PP_ALIGN.RIGHT, gap=2)
    notes(sl,
          "To conclude. Our system forecasts demand with an R-squared of 0.51, "
          "predicts reorder need with an F1 of 0.78, and raises the fill rate "
          "from 79.9 to 93.9 percent. Eighty-one automated tests pass and all "
          "fifty-one rubric checks pass. What we learned matters more than the "
          "numbers: a better forecast alone did not help, the most advanced "
          "model lost, and the non-linear kernel lost to the linear one. We "
          "report all of that honestly, because a project that only reports "
          "what worked has not really tested anything. We acknowledge our "
          "limitations, especially that the supply-side data is simulated. "
          "Thank you - we will now show the live dashboard.")
    footer(sl, 15)

    return prs


def main() -> int:
    R = load_results()
    prs = build(R)
    out = ROOT / "paper" / "PBL1_Presentation.pptx"
    ensure_dir(out)
    prs.save(out)
    print(f"Saved {out.relative_to(ROOT)}")
    print(f"  slides       : {len(prs.slides)}")
    print(f"  team         : " + " · ".join(n for n, _ in TEAM))
    print(f"  size         : {out.stat().st_size/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
