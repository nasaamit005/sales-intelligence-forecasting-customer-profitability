"""Automated reporting: multi-sheet Excel workbook and a printable HTML management report."""
from __future__ import annotations

import datetime as dt
import html

import pandas as pd

from . import config

DATA_DICTIONARY = pd.DataFrame([
    ("order_key", "Order ID + order date + customer ID", "Order IDs are re-used in the source, so this defines a distinct order."),
    ("sales / revenue", "Sum of line 'Sales' (USD)", "Net of discount; excludes shipping."),
    ("profit", "Sum of source 'Profit' field", "REPORTED profit; treated as contribution, not audited net profit."),
    ("margin", "profit / revenue", "Reported-profit margin."),
    ("aov", "revenue / distinct orders", "Average order value."),
    ("shipping_cost", "Source 'Shipping Cost'", "Not subtracted from profit (unknown whether already included)."),
    ("discount_band", "Binned line discount rate", "0%, 1-10%, 11-20%, 21-30%, 31-50%, >50%."),
    ("RFM", "Recency (days to snapshot), Frequency (orders), Monetary (revenue)", "Quintile scores 1-5; snapshot = day after last order."),
    ("historical customer value", "Revenue / reported profit per customer to date", "Backward-looking. NOT a forward CLV."),
    ("WAPE", "sum|actual-forecast| / sum|actual|", "Primary forecast accuracy metric."),
    ("Forecast interval", "Empirical 10th-90th percentile of actual/forecast ratios in validation folds", "Approximate 80% band."),
], columns=["term", "definition", "notes"])

CATALOG = pd.DataFrame([
    ("Excel workbook", "reports/excel/sales_intelligence_report.xlsx", "17-tab analytical workbook"),
    ("HTML management report", "reports/html/management_report.html", "Print to PDF from a browser"),
    ("Flask dashboard", "app.py", "Interactive dashboard (python app.py)"),
    ("Model results", "results/*.csv, results/metrics.json", "Forecast and churn evaluation"),
    ("SQLite database", "database/salesintel.db", "Star schema + forecast fact table"),
], columns=["deliverable", "location", "description"])


def _write(wb, ws_name, df, writer, title, note=None, fmt=None, index=False):
    df = df.copy()
    for c in df.columns:
        if str(df[c].dtype).startswith("period") or df[c].dtype == "object" and df[c].map(lambda x: hasattr(x, "n")).any():
            df[c] = df[c].astype(str)
    ws_name = ws_name[:31]
    df.to_excel(writer, sheet_name=ws_name, startrow=3, index=index)
    ws = writer.sheets[ws_name]
    ws.write(0, 0, title, fmt["title"])
    if note:
        ws.write(1, 0, note, fmt["note"])
    for i, c in enumerate(df.columns):
        w = max(len(str(c)), int(df[c].astype(str).str.len().quantile(0.9)) if len(df) else 8)
        low = str(c).lower()
        f = fmt["pct"] if any(k in low for k in ("margin", "share", "growth", "wape", "mape", "bias", "discount", "rate", "auc", "prob")) \
            and pd.api.types.is_float_dtype(df[c]) and df[c].abs().max() <= 5 else (
            fmt["num"] if pd.api.types.is_float_dtype(df[c]) else None)
        ws.set_column(i, i, min(max(w + 2, 10), 42), f)
    ws.freeze_panes(4, 0)
    return ws


def build_excel(t: dict, path=None):
    path = path or config.EXCEL_DIR / "sales_intelligence_report.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as xw:
        wb = xw.book
        fmt = {"title": wb.add_format({"bold": True, "font_size": 14, "font_color": "#1F3864"}),
               "note": wb.add_format({"italic": True, "font_color": "#595959"}),
               "pct": wb.add_format({"num_format": "0.0%"}), "num": wb.add_format({"num_format": "#,##0.00"}),
               "kpi_l": wb.add_format({"bold": True, "bg_color": "#E8EEF7", "border": 1}),
               "kpi_v": wb.add_format({"num_format": "#,##0", "border": 1}),
               "kpi_p": wb.add_format({"num_format": "0.0%", "border": 1})}
        k = t["kpis"]
        ws = wb.add_worksheet("Executive Dashboard")
        xw.sheets["Executive Dashboard"] = ws
        ws.write(0, 0, "Executive Dashboard: Global Superstore sales 2011-2014", fmt["title"])
        ws.write(1, 0, t["scope_note"], fmt["note"])
        for i, (lab, key, f) in enumerate([("Revenue (USD)", "revenue", "kpi_v"), ("Reported profit (USD)", "profit", "kpi_v"),
                                           ("Profit margin", "margin", "kpi_p"), ("Orders", "orders", "kpi_v"),
                                           ("Avg order value (USD)", "aov", "kpi_v"), ("Customers", "customers", "kpi_v")]):
            ws.write(3 + i, 0, lab, fmt["kpi_l"]); ws.write(3 + i, 1, float(k[key]), fmt[f])
        ws.set_column(0, 0, 26); ws.set_column(1, 1, 18)
        m = t["monthly"]
        _write(wb, "Monthly Sales Trends", m, xw, "Monthly sales trends", "Revenue, reported profit, orders, AOV and YoY growth by month.", fmt)
        n = len(m)
        ch = wb.add_chart({"type": "line"})
        ch.add_series({"name": "Revenue", "categories": ["Monthly Sales Trends", 4, 0, 3 + n, 0], "values": ["Monthly Sales Trends", 4, 1, 3 + n, 1]})
        ch.add_series({"name": "Profit", "categories": ["Monthly Sales Trends", 4, 0, 3 + n, 0], "values": ["Monthly Sales Trends", 4, 2, 3 + n, 2]})
        ch.set_title({"name": "Monthly revenue and reported profit"}); ch.set_size({"width": 760, "height": 320})
        ws.insert_chart("D4", ch)
        _write(wb, "Sales KPI Summary", t["yearly"], xw, "Sales KPI summary by year", "Calendar-year KPIs.", fmt)
        _write(wb, "Product Performance", t["products"], xw, "Top and bottom products by reported profit", "Top 10 and bottom 10.", fmt)
        _write(wb, "Category Analysis", t["subcat"], xw, "Category and sub-category analysis", None, fmt)
        _write(wb, "Regional Analysis", t["market"], xw, "Regional analysis by market and year", None, fmt)
        _write(wb, "Customer Segmentation", t["segments"], xw, "RFM customer segments", "Rule-based segments from quintile RFM scores.", fmt)
        _write(wb, "RFM Analysis", t["rfm"], xw, "Customer-level RFM table", None, fmt)
        _write(wb, "Cohort Retention", t["cohort"], xw, "Quarterly cohort retention", "Share of each first-purchase-quarter cohort active N quarters later.", fmt)
        _write(wb, "Customer Lifetime Value", t["cust_value"], xw, "Historical customer value", "HISTORICAL revenue/profit to date, not a predicted CLV.", fmt)
        _write(wb, "Forecast Comparison", t["selection"], xw, "Forecast model comparison (validation folds)", "Mean of 3 rolling-origin folds; selection used these only.", fmt)
        _write(wb, "Forecast Accuracy", t["forecast_results"], xw, "Forecast accuracy by fold and held-out test", "Test block was not used for model selection.", fmt)
        _write(wb, "Discount Analysis", t["discount_bands"], xw, "Discount band analysis", "Descriptive only: discounts are not randomly assigned, so no causal claim.", fmt)
        _write(wb, "Demand Planning", t["horizon"], xw, "Forecast-based demand scenarios", "Scenario summary; no inventory data exists, so no reorder points.", fmt)
        _write(wb, "Data Quality", t["dq"], xw, "Data-quality report", None, fmt)
        _write(wb, "Data Dictionary", DATA_DICTIONARY, xw, "Data dictionary and metric definitions", None, fmt)
        _write(wb, "Report Catalog", CATALOG, xw, "Report catalog", None, fmt)
    return path


def build_html(t: dict, path=None):
    path = path or config.HTML_DIR / "management_report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    k, m = t["kpis"], t["metrics"]

    def tbl(df, fmts=None):
        return df.to_html(index=False, border=0, classes="t", float_format=lambda x: f"{x:,.3f}" if abs(x) < 10 else f"{x:,.0f}")

    best = m["forecast"]["best_model"]
    html_doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Management Report</title>
<style>body{{font-family:Calibri,Arial,sans-serif;max-width:900px;margin:30px auto;color:#222;line-height:1.45}}
h1{{color:#1F3864}}h2{{color:#1F3864;border-bottom:2px solid #1F3864;padding-bottom:3px;margin-top:28px}}
.kpi{{display:flex;gap:12px;flex-wrap:wrap}}.k{{background:#E8EEF7;padding:10px 14px;border-radius:6px;min-width:130px}}
.k b{{display:block;font-size:20px}}table.t{{border-collapse:collapse;font-size:12px;width:100%}}
table.t th{{background:#1F3864;color:#fff;padding:4px 6px;text-align:left}}table.t td{{padding:3px 6px;border-bottom:1px solid #ddd}}
.note{{background:#FFF8E1;border-left:4px solid #F9A825;padding:8px 12px;font-size:13px}}</style></head><body>
<h1>Sales Intelligence: Management Report</h1>
<p>Generated {dt.date.today():%d %b %Y}. Source: Global Superstore sample data (fictional), {t['scope_note']}</p>
<div class="kpi">
<div class="k">Revenue<b>${k['revenue']:,.0f}</b></div><div class="k">Reported profit<b>${k['profit']:,.0f}</b></div>
<div class="k">Margin<b>{k['margin']:.1%}</b></div><div class="k">Orders<b>{k['orders']:,}</b></div>
<div class="k">Customers<b>{k['customers']:,}</b></div></div>
<h2>Key findings</h2><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in t['findings'])}</ul>
<h2>Forecast evaluation</h2>
<p>Selected model: <b>{best}</b> (lowest mean WAPE across 3 rolling-origin validation folds). Held-out test WAPE for all models:</p>
{tbl(t['test_table'])}
<h2>Discount bands</h2>{tbl(t['discount_bands'])}
<h2>Demand scenarios (next 13 weeks)</h2>{tbl(t['horizon'])}
<div class="note"><b>Limitations.</b> Dataset is a fictional sample. Profit is the source's reported field, not audited. No returns,
inventory, lead-time or cost data exist, so no net-profit, reorder-point or causal discount-effect claims are made.</div>
<h2>Data quality</h2>{tbl(t['dq'])}
</body></html>"""
    path.write_text(html_doc, encoding="utf-8")
    return path
