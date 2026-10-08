# ERP Integration Readiness

## What is and is not claimed

**Not implemented:** this project does not connect to ERPNext, SAP, Odoo or any
other ERP. There is no connector, no API client, no authentication, and no
purchase order is created in any external system.

**What "in ERP Systems" means in the title:** the decision-support component is
designed against an ERP-shaped data model, so that connecting it later is a
mapping exercise rather than a redesign. That claim is testable — the entity
model below is the one the code actually uses.

---

## 1. The ERP-style entity model used

| Entity | Fields used | Where it lives now | Real or simulated |
|---|---|---|---|
| **Item** | `stock_code`, `description`, `median_price` | `data/processed/sku_master.parquet` | Real |
| **Sales History** | `date`, `stock_code`, `demand`, `revenue`, `avg_price` | `daily_demand_panel.parquet` | Real |
| **Warehouse / Bin** | implicit single location | — | Single-echelon assumption |
| **Stock Ledger** | `beginning_stock`, `arrivals`, `consumed`, `ending_stock` | `inventory_state.parquet` | **SIMULATED** |
| **Supplier** | `supplier_id`, `lead_time_days`, `reliability` | `suppliers.parquet` | **SIMULATED** |
| **Supplier Item** | `unit_cost`, `moq`, `pack_size`, `capacity_units` | `suppliers.parquet` | **SIMULATED** |
| **Purchase Order (draft)** | `rec_qty`, `rec_order_value`, `supplier_id` | `procurement_recommendations.parquet` | Output |

These names deliberately track ERPNext doctypes (`Item`, `Bin`,
`Stock Ledger Entry`, `Supplier`, `Item Supplier`, `Purchase Order`) so the
mapping is close to one-to-one.

---

## 2. What a real integration would require

### Inputs the ERP must supply (replacing the simulator)

```
Item               -> item_code, item_name, stock_uom
Bin                -> actual_qty, ordered_qty, warehouse      [replaces current_stock, on_order]
Stock Ledger Entry -> posting_date, actual_qty                [replaces demand history if sales come from here]
Supplier           -> supplier_name, lead_time_days
Item Supplier      -> min_order_qty, conversion_factor, price_list_rate
```

The moment these are available, `sim/` is deleted and
`src/xaierp/policy/reorder.py` is unchanged — it already takes
`inventory_position`, `lead_time_days`, `moq`, `pack_size`,
`supplier_capacity` and `unit_cost` as plain arguments.

### Output the system would write back

A **draft** Purchase Order per supplier, with lines carrying the recommended
quantity and the generated explanation in the remarks field.

### Sketch of an ERPNext connector

```python
# NOT IMPLEMENTED -- this is the shape such a connector would take.
import requests

class ERPNextClient:
    def __init__(self, base_url: str, api_key: str, api_secret: str):
        self.base = base_url.rstrip("/")
        self.headers = {"Authorization": f"token {api_key}:{api_secret}"}

    def get_stock(self, item_code: str, warehouse: str) -> dict:
        r = requests.get(
            f"{self.base}/api/resource/Bin",
            params={"filters": f'[["item_code","=","{item_code}"],'
                               f'["warehouse","=","{warehouse}"]]',
                    "fields": '["actual_qty","ordered_qty"]'},
            headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()["data"][0]

    def create_draft_po(self, supplier: str, items: list[dict], remarks: str):
        # docstatus=0 means DRAFT. Never submit (docstatus=1) automatically.
        payload = {"supplier": supplier, "docstatus": 0,
                   "remarks": remarks, "items": items}
        r = requests.post(f"{self.base}/api/resource/Purchase Order",
                          json=payload, headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()["data"]["name"]
```

---

## 3. Non-negotiable safety rule

**A purchase order must never be auto-submitted.**

The system produces a *recommendation*. A human buyer approves it. Three
reasons, all of which this project's own results support:

1. The forecast is wrong often enough to matter — test RMSE is 288 units
   against a mean 7-day demand of 200 units.
2. The supply-side parameters that size the order are, in this build, entirely
   simulated.
3. The classifier's recall is 0.816, meaning roughly 18% of genuine reorder
   situations are missed. A human reviewing the queue catches those.

Hence `docstatus=0` in the sketch above, and hence the explanation layer: the
point of generating a reason for every recommendation is to make human review
fast enough to actually happen.

---

## 4. Integration readiness checklist

| Requirement | Status |
|---|---|
| Decision logic takes ERP fields as plain arguments | **Ready** — `decide_reorder()` |
| No ERP-specific types leak into the model layer | **Ready** |
| Supplier constraints (MOQ, pack, capacity) modelled | **Ready** |
| Multi-supplier selection | **Ready** — `is_preferred`, cheapest-first with lead-time tie-break |
| Explanation attached to every recommendation | **Ready** |
| Human-in-the-loop by construction | **Ready** — draft only |
| Authentication / secrets handling | Not implemented |
| Write-back connector | Not implemented |
| Multi-warehouse / multi-echelon | Not implemented — single location assumed |
| Currency and UOM conversion | Not implemented |
| Incremental / scheduled runs | Not implemented — batch only |

Six of eleven ready. The five gaps are all integration plumbing rather than
modelling work, which is the intended position for PBL-1.
