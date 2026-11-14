# Data source and provenance

* **File:** `data/raw/superstore_global_2011_2014.csv` (51,290 rows x 24 columns), unmodified.
* **Obtained from:** a public GitHub mirror (`pplonski/datasets-for-start`, path
  `superstore-sales/superstore_dataset2011-2015.csv`). The file contains orders dated 2011-01-01 to 2014-12-31
  despite the "2015" in the upstream file name.
* **Nature:** the "Global Superstore" sample dataset distributed with Tableau training material. It is
  **fictional**; the original publisher's licence terms were not independently verified. Confirm the licence
  before any redistribution or commercial use, and do not present results as real company performance.
* **Not available in this data:** returns/cancellations, product cost, inventory levels, lead times,
  promotions calendar. The analysis is therefore limited to what the fields can support (see README, section 9).
* **Other datasets** discussed in the planning document (UCI Online Retail II, Olist) were *not* used:
  they were not reachable from the build environment. The code does not depend on them.

## Known source quirks handled in `data.py`
| Issue | Evidence | Handling |
|---|---|---|
| Mixed date formats (`dd-mm-yyyy` and `d/m/yyyy`) | Day-first parsing gives ship lags of 0-7 days; month-first gives thousands of negative lags | `parse_mixed_dates` |
| Order IDs re-used across dates/customers | 659 IDs map to several customers/dates | `order_key = order_id + date + customer_id` (25,754 orders vs 25,035 IDs) |
| Customer names shared by several IDs | 795 names | `customer_id` is the customer key |
| Product IDs with several names | 457 IDs | `product_key = product_id + name` |
| Postal code mostly missing | 80.5% | Not used |
