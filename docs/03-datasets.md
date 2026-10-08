# 03 — Dataset Audit (verified)

> Every numeric claim below is tagged with a verification status.
> ✅ = confirmed at the primary source (UCI / Kaggle) or by arithmetic.
> ⚠️ = reported by secondary sources; confirm yourself before it goes in the paper.

---

## 1. Summary of the decision

| Role | Dataset | Why |
|------|---------|-----|
| **Primary** | Corporación Favorita — Kaggle *Store Sales – Time Series Forecasting* | Clean, tidy, rich exogenous covariates (promotions, holidays, oil, store metadata). Best forecasting environment. |
| **Primary, SKU-level** | Corporación Favorita — original 2017 competition release | Item-level granularity. **Required** if you want to make any claim about intermittent/lumpy demand. |
| **External validation** | UCI Online Retail II | Completely different business (UK online gift wholesaler vs Ecuadorian grocery). Tests generalisation, not just tuning. |
| **Simulated overlay** | Supply-side parameters (suppliers, lead times, costs, MOQ, capacity) | **Does not exist in any public dataset.** Must be generated and declared. |

The two real datasets are **not merged**. They are separate experimental tracks. Merging retail transaction data from two unrelated businesses produces a meaningless combined series and would be a methodological error.

---

## 2. Corporación Favorita — Kaggle *Store Sales* release

**URL:** https://www.kaggle.com/competitions/store-sales-time-series-forecasting

### 2.1 Files and schema

| File | Columns | Notes |
|---|---|---|
| `train.csv` | `id`, `date`, `store_nbr`, `family`, `sales`, `onpromotion` | Target = `sales`. **Fractional values are valid** (products sold by weight). `onpromotion` = count of items in that family on promotion at that store that day. ✅ |
| `test.csv` | `id`, `date`, `store_nbr`, `family`, `onpromotion` | Same features minus target. ✅ |
| `stores.csv` | `store_nbr`, `city`, `state`, `type`, `cluster` | `cluster` is a grouping of similar stores. ✅ |
| `oil.csv` | `date`, `dcoilwtico` | Daily WTI oil price. Ecuador's economy is oil-dependent. ✅ |
| `holidays_events.csv` | `date`, `type`, `locale`, `locale_name`, `description`, `transferred` | ✅ |
| `transactions.csv` | `date`, `store_nbr`, `transactions` | Store footfall proxy. Training timeframe only. ✅ |
| `sample_submission.csv` | `id`, `sales` | — |

### 2.2 Verified dimensions

| Fact | Value | Status |
|---|---|---|
| `train.csv` rows | **3,000,888** | ✅ |
| Stores | **54** | ✅ |
| Product families | **33** | ✅ |
| Distinct series (store × family) | **1,782** | ✅ (54 × 33) |
| Training date range | **2013-01-01 → 2017-08-15** | ✅ |
| Distinct dates present in train | **1,684** | ✅ by arithmetic: 3,000,888 ÷ 1,782 = 1,684 |
| Calendar days in that span | 1,688 | ✅ |
| **Missing dates** | **4** — the panel is *not* fully dense | ✅ derived: 1,688 − 1,684 = 4 |
| Test horizon | 2017-08-16 → 2017-08-31 (**16 days**, 28,512 rows = 1,782 × 16) | ⚠️ Kaggle's prose says "15 days after the last date"; the row count implies 16. **Verify by loading the file.** Report what you measure, not what the prose says. |
| Competition metric | RMSLE | ⚠️ verify on the Evaluation tab |

> **Action item:** the 4 missing dates are the store-closure days (25 December in each of 2013–2016). Confirm this by a `set` difference on dates in your first EDA notebook and document it. A model that silently reindexes to a dense calendar will inject spurious zeros on Christmas — a real, findable bug.

### 2.3 The critical caveat — aggregation level

`sales` here is **total sales per product *family* per store per day**. `family` has 33 values (`GROCERY I`, `BEVERAGES`, `PRODUCE`, `CLEANING`, `DAIRY`, `BREAD/BAKERY`, `POULTRY`, `MEATS`, `PERSONAL CARE`, `HOME CARE`, `DELI`, `EGGS`, `FROZEN FOODS`, `LIQUOR,WINE,BEER`, `PREPARED FOODS`, `AUTOMOTIVE`, `BEAUTY`, `BOOKS`, `CELEBRATION`, `HARDWARE`, `HOME AND KITCHEN I/II`, `HOME APPLIANCES`, `LADIESWEAR`, `LAWN AND GARDEN`, `LINGERIE`, `MAGAZINES`, `PET SUPPLIES`, `PLAYERS AND ELECTRONICS`, `SCHOOL AND OFFICE SUPPLIES`, `SEAFOOD`, `BABY CARE`, `GROCERY II` — verify exact list).

**Consequences:**

1. Family-level aggregation **smooths away intermittency**. Aggregate GROCERY I sales at a store are near-continuous. A few small families (`BOOKS`, `BABY CARE`, `HOME APPLIANCES`) will still be sparse — those are your intermittent examples, but there will not be many.
2. **You cannot honestly answer RQ1 ("which model wins under intermittent demand") on this file alone.** Run the SBC classification (ADI, CV²) in EDA and report the actual class distribution. If fewer than ~15% of series fall into intermittent/lumpy, either (a) move to the SKU-level release, or (b) rewrite RQ1 to be about *volume/volatility strata* instead of intermittency. Option (b) is legitimate; silently claiming intermittency is not.
3. Family-level replenishment is not how procurement works — you buy SKUs, not categories. State this as a modelling simplification, or use the SKU-level release for the procurement experiments.

### 2.4 Known data-engineering issues

- `oil.csv` has **gaps on weekends and holidays** (no trading). Forward-fill, and document the choice.
- `holidays_events.csv`: rows with `transferred = True` mean the holiday was *officially* on that date but *celebrated* on another; a separate row of `type = Transfer` carries the actual celebration date. `type = Bridge` are extra days off; `type = Work Day` are compensating working days (often Saturdays). `locale` ∈ {National, Regional, Local} must be joined on `locale_name` against `stores.state` / `stores.city` — a naive date-only join over-applies local holidays to all 54 stores. **This join is the single most common bug in public notebooks on this dataset.**
- Known structural events to include as features: the **2016-04-16 Ecuador earthquake** (large donation-driven demand spike in the following weeks), payday effects (the 15th and last day of each month), and the fact that some store × family series **start late** (store opened, or family introduced) — leading zeros are structural, not demand.

---

## 3. Corporación Favorita — original 2017 competition release (SKU level)

**URL:** https://www.kaggle.com/c/favorita-grocery-sales-forecasting

| Fact | Value | Status |
|---|---|---|
| Training observations | **125,497,040** | ⚠️ widely reported; verify |
| Test observations | **3,370,464** | ⚠️ |
| Stores | 54, across 22 cities in 16 states of Ecuador | ⚠️ |
| Items | ~4,000–4,400 items in 33 families and **337 classes** | ⚠️ sources differ on the item count — count `items.csv` yourself |
| Extra fields vs playground release | `item_nbr`, `items.perishable` (0/1), `items.class`, `unit_sales` (can be **negative** = returns) | ⚠️ |
| Metric | NWRMSLE, with perishable items weighted higher than non-perishable | ⚠️ verify the exact weight on the Evaluation tab |

**Why you may want it:**
- Real SKU granularity → genuine intermittent and lumpy series → RQ1 has teeth.
- The `perishable` flag gives you a **data-grounded asymmetric cost ratio** (perishables have a much higher overage cost) instead of an invented one. This is a small but genuinely good methodological detail for the paper.
- 337 classes gives a hierarchy (family → class → item) if you want hierarchical reconciliation as an extension.

**Why it may not be worth it:**
- 125M rows. On a laptop this needs `polars` or `duckdb` and careful dtype management (`int8`/`int16`/`float32`), or a stratified subsample.
- **Recommended compromise:** take a *stratified random sample* of ~300–500 items spanning all four SBC classes and all 33 families, across all 54 stores. Document the sampling procedure precisely. This is defensible, reproducible, and runs on modest hardware.

---

## 4. UCI Online Retail II

**URL:** https://archive.ics.uci.edu/dataset/502/online+retail+ii

| Fact | Value | Status |
|---|---|---|
| Instances | **1,067,371** | ✅ |
| Features | **8** | ✅ |
| Date range | **2009-12-01 → 2011-12-09** | ✅ |
| Donor | **Daqing Chen (2012)** | ✅ |
| Licence | **CC BY 4.0** | ✅ |
| DOI | **10.24432/C5CG6D** | ✅ |
| Missing values | **Yes** | ✅ |
| Business | UK-based, registered, non-store **online retailer of all-occasion giftware**; many customers are **wholesalers** | ✅ |

### 4.1 Schema

| Attribute | Type | Meaning |
|---|---|---|
| `InvoiceNo` | Nominal | 6-digit; **prefix `C` = cancellation** |
| `StockCode` | Nominal | 5-digit product code (with non-conforming exceptions, see below) |
| `Description` | Nominal | Product name |
| `Quantity` | Numeric | Units per transaction line; **negative on returns/cancellations** |
| `InvoiceDate` | Datetime | Timestamp |
| `UnitPrice` | Numeric | Price per unit, GBP (£) |
| `CustomerID` | Nominal | 5-digit; **substantially missing** |
| `Country` | Nominal | Customer country |

> The Kaggle mirror (`mashlyn/online-retail-ii-uci`) renames some columns: `Invoice`, `Price`, `Customer ID`. The UCI original ships as a `.xlsx` with **two sheets** — `Year 2009-2010` and `Year 2010-2011` — which must be concatenated. Handle both naming conventions in your loader.

### 4.2 Cleaning protocol (write this down; it goes in the paper)

1. Concatenate both sheets.
2. Drop exact duplicate rows (there are some).
3. **Cancellations:** rows where `InvoiceNo` starts with `C`. Two valid treatments — (a) drop them, or (b) net them against the original sale. Choose (b) if you want *true* demand; choose (a) for simplicity. **Document which, and be consistent.** Returns matter here because ~1–2% of lines are cancellations and wholesale returns are large.
4. **Non-product stock codes** must be removed: `POST`, `DOT` (postage), `D` (discount), `M`/`m` (manual), `C2` (carriage), `S` (samples), `BANK CHARGES`, `AMAZONFEE`, `PADS`, `CRUK` (charity), `TEST001/2`, `ADJUST`/`ADJUST2` (adjustments), `gift_0001_*`. Filter by regex `^\d{5}[A-Za-z]?$` and manually inspect the residue rather than trusting a hard-coded list.
5. `UnitPrice ≤ 0` and `Quantity ≤ 0` (after cancellation handling) → inspect and drop.
6. `CustomerID` missing on a large minority of rows — **irrelevant for this project**, since you are forecasting per-product demand, not per-customer. Do not drop rows for missing `CustomerID`; that is a CLV/RFM convention and would silently discard real demand.
7. **Aggregate to a daily (or weekly) product demand series:** `groupby(StockCode, date).Quantity.sum()`. Restrict to the UK (~90%+ of rows) or model countries separately — mixing them creates a lumpy multi-market series.
8. Filter to products with sufficient history (e.g. ≥ 52 weeks of presence and ≥ 30 non-zero demand days) and record how many products survive.

### 4.3 Why this dataset is genuinely useful here

It is **structurally hostile** in ways Favorita is not, and that is the point of an external-validation set:

- No promotions, no holidays file, no store metadata, no covariates at all beyond price and calendar.
- Wholesale order patterns → **highly lumpy, high-CV² demand**. This is where Croston/TSB/SBA baselines will actually matter, and where your intermittency claims can live even if you stay with family-level Favorita.
- Short history (2 years), heavy product churn.
- Aggressive seasonality around Christmas (a giftware retailer).

If your framework holds up on both a smooth grocery panel and a lumpy wholesale gift catalogue, the generalisation claim is real.

---

## 5. What neither dataset contains — and what you must do about it

### 5.1 The honest gap table

| Field needed by the procurement layer | Favorita | Online Retail II | Source |
|---|---|---|---|
| Historical sales / demand | ✅ | ✅ | real |
| Calendar, holidays, promotions | ✅ | ✗ | real |
| Price | ✗ (not in the playground release) | ✅ | real |
| Product hierarchy | ✅ family/class | partial (description text) | real |
| **On-hand inventory position** | ✗ | ✗ | **simulated** |
| **Supplier identity** | ✗ | ✗ | **simulated** |
| **Supplier lead time (mean & variance)** | ✗ | ✗ | **simulated** |
| **Supplier reliability / fill rate** | ✗ | ✗ | **simulated** |
| **Purchase orders / receipts** | ✗ | ✗ | **simulated** |
| **Unit purchase cost, MOQ, case pack** | ✗ | ✗ | **simulated** |
| **Holding & shortage cost rates** | ✗ | ✗ | **simulated** |
| **Supplier capacity, budget** | ✗ | ✗ | **simulated** |

There is no public dataset that contains the right-hand column joined to real sales. Procurement terms are commercially confidential. **Stop looking for one** — the search cost is unbounded and the outcome is known.

### 5.2 Rules for handling the simulation (non-negotiable)

1. Call it **"simulated supply-side parameters"** everywhere — abstract, methodology, results, figure captions, limitations.
2. Never present a simulated number as an observation.
3. **Parameterise, then sweep.** A single hand-picked parameter set proves nothing. Four scenarios (see [06](06-experimental-design.md) §3) turn the weakness into a sensitivity analysis, which is a strength.
4. Publish the generator with a fixed seed. Reproducibility is the mitigation.
5. Ground parameters in cited sources where possible — e.g. annual holding cost as a percentage of item value, standard service-level targets — rather than round numbers pulled from air.

### 5.3 The simulation layer, specified

Inventory state is **initialised** and then evolved by the digital twin (see [04](04-methodology.md) §7); it is not sampled independently per period.

Supplier attributes, per item, drawn once with a fixed seed:

| Parameter | Distribution / rule | Rationale |
|---|---|---|
| Suppliers per item | 2–3 | Enough to make allocation non-trivial |
| Lead time L | `Gamma(k, θ)` truncated to ≥ 1 day, per-supplier mean 3–14 days | Gamma is right-skewed and non-negative — realistic for lead times; Normal is not |
| Lead-time CV | 0.15 (reliable) to 0.50 (unreliable) | Defines the reliability spread |
| On-time delivery rate | 0.80–0.98 | Drives late-delivery events |
| Unit cost c | Base cost × (0.90–1.10); **cheaper suppliers get longer/more variable lead times** | Creates a genuine cost-vs-risk trade-off. Without this correlation the optimizer's job is trivial and the paper is boring. |
| MOQ | {1, 10, 50, 100, 250} by item volume decile | Forces integrality; makes MILP necessary rather than decorative |
| Case pack / order multiple | {1, 6, 12, 24} | Creates the rounding term in the decision decomposition |
| Supplier capacity per period | 1.5–4× mean demand | Occasionally binding |
| Quantity discount | Optional price break at 2× MOQ, 3–7% off | Adds a piecewise-linear cost term |
| Holding cost rate h | 25% of unit cost per annum, pro-rated daily | Standard carrying-cost range from the inventory literature |
| Shortage cost rate p | `ratio` × h, ratio ∈ {2, 5, 9, 20} → implied CSL ∈ {0.67, 0.83, 0.90, 0.95} | Directly sets the newsvendor critical ratio; lets you sweep service targets meaningfully |
| Perishable items (Favorita SKU release only) | Overage cost = full unit cost (write-off), not just carrying cost | Grounded in the real `perishable` flag |
| Fixed ordering cost K | Per PO, per supplier | Makes order *frequency* a real decision |

**Deliverable:** `sim/supply_config.yaml` + `sim/generate_supply.py`, both released. Every number above lives in the YAML, not in code.

---

## 6. Acquisition

```bash
pip install kaggle ucimlrepo

# --- Favorita (playground release) ---
# Requires a Kaggle account + accepting the competition rules once in the browser,
# and ~/.kaggle/kaggle.json  (Windows: C:\Users\<you>\.kaggle\kaggle.json)
kaggle competitions download -c store-sales-time-series-forecasting -p data/raw/favorita
# ~20 MB zipped

# --- Favorita (original SKU-level release) ---
kaggle competitions download -c favorita-grocery-sales-forecasting -p data/raw/favorita_sku
# ~470 MB compressed (7z); expands to several GB. Use polars/duckdb.

# --- UCI Online Retail II ---
python -c "from ucimlrepo import fetch_ucirepo; d=fetch_ucirepo(id=502); d.data.original.to_parquet('data/raw/online_retail_ii.parquet')"
# or download online_retail_II.xlsx directly (two sheets) from archive.ics.uci.edu
```

### Licensing and citation obligations

- **Online Retail II** — CC BY 4.0. Attribution is *required*. Cite: Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
- **Favorita (Kaggle competitions)** — governed by the competition rules accepted at download. Kaggle's standard terms permit use of competition data for academic research and education, but **prohibit redistribution** to non-participants. Therefore: **do not commit the raw data to your GitHub repository.** Ship download scripts and checksums instead. Cite the competition page and Corporación Favorita as the data provider.
- Your **simulated supply layer** is your own work — release it under a permissive licence (MIT / CC BY 4.0). That released artefact is a small, real contribution in its own right and worth a sentence in the paper.

---

## 7. EDA checklist (before any modelling)

Produce these in `notebooks/01_eda.ipynb` and keep the outputs — several become paper figures.

- [ ] Row/column counts, dtypes, memory footprint; confirm every ✅/⚠️ figure in this document
- [ ] Date completeness per series; identify and name the 4 missing Favorita dates
- [ ] Distribution of series lengths; count series with structural leading zeros (late starts)
- [ ] **SBC classification**: compute ADI and CV² per series; plot the ADI–CV² scatter with the 1.32 / 0.49 cutoffs; report the 2×2 class counts for *both* datasets → **paper figure**
- [ ] Zero-demand proportion per series; histogram
- [ ] Seasonality decomposition: weekly, monthly, annual; payday effect (15th / month-end)
- [ ] Promotion lift analysis (`onpromotion` vs sales) → justifies including it as a feature
- [ ] Holiday effect by `locale`, done with the correct locale join
- [ ] Earthquake window (2016-04-16 onward) — quantify the anomaly and decide: feature flag, or excluded window? Document it.
- [ ] Oil price vs aggregate sales correlation — if it is negligible, *say so* and drop the feature; a negative result honestly reported is better than a useless feature carried for show
- [ ] Online Retail II: cancellation rate, non-product stock-code inventory, country distribution, product churn rate
- [ ] Price statistics and volume deciles → these drive the simulated MOQ and cost assignment
