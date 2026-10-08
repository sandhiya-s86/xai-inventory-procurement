# Implementation Audit

State of the repository before implementation began, and the disposition of
each existing component.

## Starting state

The repository was a **proposal / pre-implementation specification**. It
contained 14 Markdown documents and a README. There was:

- no source code
- no tests
- no notebooks
- no data, and no way to obtain any
- no results, figures or tables
- no dependency manifest or environment definition

Total: 15 files, all prose.

## Disposition of existing files

| Existing component | State found | Action | Reason |
|---|---|---|---|
| `README.md` | Research-proposal framing | **Modify** | Rewritten as a usable project README with real results and run instructions. Original research framing preserved in `docs/`. |
| `docs/00-executive-summary.md` | Prose | **Keep** | Still an accurate statement of the research ambition. |
| `docs/01-problem-and-motivation.md` | Prose | **Keep** | Motivation is unchanged by the scope cut. |
| `docs/02-literature-review-and-research-gap.md` | Prose | **Keep** | Useful for the paper; not needed for PBL-1 marks. |
| `docs/03-datasets.md` | Proposed several datasets | **Supersede** | Superseded by `docs/dataset_acquisition.md`, which records the dataset actually used and why. Original kept for provenance. |
| `docs/04-methodology.md` | Research-scale methodology | **Supersede** | Superseded by `docs/pbl1_scope.md`. Original kept. |
| `docs/05-system-architecture.md` | Proposed architecture | **Keep** | Still describes the target system; the implemented subset is a faithful slice of it. |
| `docs/06-experimental-design.md` | Proposed experiments | **Supersede** | Superseded by the implemented experiment set. Original kept. |
| `docs/07-evaluation-metrics.md` | Metric definitions | **Keep** | Metric choices implemented as specified. |
| `docs/08-implementation-plan.md` | Plan | **Keep** | Historical record of intent. |
| `docs/09-risks-and-mitigations.md` | Risk register | **Keep** | The data-honesty risk it flags is addressed by `sim/`. |
| `docs/10-paper-outline.md` | Outline | **Keep** | Basis for `paper/pbl1_report.md`. |
| `docs/11-references.md` | References | **Keep** | — |
| `docs/decisions.md` | Decision log | **Keep** | Appended to rather than replaced. |
| `docs/lit-log.md` | Literature log | **Keep** | — |

**No original document was deleted or overwritten.** Documents that the
implementation superseded are explicitly marked as superseded in this table,
with the replacement named.

## What was added

| Area | Added |
|---|---|
| Environment | `requirements.txt`, `pyproject.toml`, `.gitignore`, `pytest.ini` |
| Config | `config/project.yaml`, `sim/supply_config.yaml` |
| Source | `src/xaierp/` — 8 subpackages, ingest → features → models → policy → simulate → explain → utils |
| Simulation | `sim/generate_supply.py`, `sim/inventory_state.py` |
| Scripts | 9 scripts: download, prepare, EDA, features, train (5 stages), simulate, explain, plots, pipeline, completeness, notebooks |
| Notebooks | 10, all executed |
| Tests | 7 files, 81 tests |
| Dashboard | `dashboard/app.py`, 6 pages |
| Docs | scope, rubric mapping, dataset acquisition, EDA insights, model selection, ERP readiness, viva, presentation, this audit |
| Report | `paper/pbl1_report.md` |

## Gaps found in the proposal, and how they were resolved

| Gap | Resolution |
|---|---|
| Proposal assumed inventory and supplier data would exist. **No public sales dataset has it.** | Built an explicit, seeded, documented simulator and labelled every output as SIMULATED. The alternative — quietly inventing numbers and calling them data — was not acceptable. |
| Proposal named Favorita as primary but Kaggle needs credentials. | Switched to UCI Online Retail II (no credentials) and documented the trade-off in `docs/dataset_acquisition.md`. |
| Proposal listed TFT / DeepAR / N-BEATS / MILP / digital twin. | Deferred, with reasons, in `docs/pbl1_scope.md §2`. |
| Proposal had no leakage strategy for a time-series target. | Defined an explicit timing convention and enforced it with 8 dedicated tests. |
| Proposal had no way to verify explanations were truthful. | Built the decision decomposition as an exact identity and gated it with tests. |

## Environment notes

Developed on Windows 11, Python 3.11, in a machine-constrained setting
(16 GB RAM with ~46 GB of 49 GB commit charge already in use by other
software). Two accommodations are baked into the code and are worth knowing
about:

1. **The 45 MB `.xlsx` is streamed, not loaded.** `pandas.read_excel`
   materialises the whole sheet plus openpyxl's object graph and failed with
   `_ArrayMemoryError`. `ingest/online_retail.py::_stream_sheet` iterates rows
   in read-only mode and writes Parquet in 50k-row chunks.
2. **Training is split into one subprocess per stage** and BLAS thread pools
   are capped to 1. Holding every fitted model and design matrix in one
   interpreter triggered `OpenBLAS: Memory allocation still failed`.

Neither accommodation changes any result; both are documented where they occur.
