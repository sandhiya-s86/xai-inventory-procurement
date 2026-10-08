"""xaierp -- Explainable AI decision support for inventory and procurement.

Package layout
--------------
ingest/    read + validate + clean the raw Online Retail II workbook
features/  calendar, lag and rolling features (no future information)
models/    regression and classification model zoo
policy/    reorder quantity logic (safety stock, MOQ, pack rounding)
simulate/  inventory and procurement cost simulation
explain/   SHAP and decision-level explanations
evaluate/  model comparison tables and plots
utils/     config, seeding, metrics
"""

__version__ = "1.0.0"
