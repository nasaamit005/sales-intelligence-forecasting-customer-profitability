"""Routes and chart builders. Every number on a page is computed from the pipeline outputs, never hard-coded."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from flask import Blueprint, Response, abort, current_app, render_template, request, send_file

from salesintel import analytics as A
from salesintel import commercial as C
from salesintel import config, data

bp = Blueprint("main", __name__)
PALETTE = ["#1F3864", "#2E75B6", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#9C755F"]
NAV = [("overview", "Executive Overview"), ("sales", "Sales Analysis"), ("products", "Product Intelligence"),
       ("regions", "Regional Intelligence"), ("customers", "Customer Intelligence"), ("forecasting", "Forecasting"),
       ("discounts", "Pricing & Discounts"), ("planning", "Demand Planning"), ("reports", "Reports")]
FILTERABLE = {"overview", "sales", "products", "regions", "customers", "discounts"}


@dataclass
class DataStore:
    df: pd.DataFrame
    r: dict

    @classmethod
    def load(cls):
        R = config.RESULTS_DIR
        r = {"metrics": json.loads((R / "metrics.json").read_text()),
             "weekly": pd.read_csv(R / "weekly_revenue.csv", parse_dates=["week_ending"]),
             "preds": pd.read_csv(R / "forecast_predictions.csv", parse_dates=["forecast_date"]),
             "future": pd.read_csv(R / "forecast_next13.csv", parse_dates=["week_ending"]),
             "plan": pd.read_csv(R / "demand_plan.csv", parse_dates=["week_ending"]),
             "selection": pd.read_csv(R / "model_selection.csv"),
             "fc_results": pd.read_csv(R / "forecast_evaluation.csv"),
             "cat_eval": pd.read_csv(R / "category_forecast_evaluation.csv"),
             "churn": pd.read_csv(R / "churn_scores_2014H2.csv"),
             "weak": pd.read_csv(R / "weak_profit_products.csv")}
        return cls(data.load_clean(), r)


# ------------------------------- helpers -------------------------------
def style(fig, h=360):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=50, r=20, t=40, b=40), colorway=PALETTE,
                      font=dict(family="Segoe UI, Calibri, Arial", size=12), legend=dict(orientation="h", y=-0.2),
                      title_font=dict(size=14, color="#1F3864"))
    return fig


def chart(fig, title, note=None, wide=False):
    fig.update_layout(title=title)
    return {"type": "chart", "json": pio.to_json(style(fig)), "note": note, "wide": wide}


def table(df, name, title, note=None, n=15, filters=True):
    shown = df.head(n)
    return {"type": "table", "title": title, "html": fmt_table(shown), "name": name, "note": note, "filters": filters,
            "rows": len(df)}


PCT_KEYS = ("margin", "share", "growth", "rate", "wape", "mape", "bias", "discount", "probability", "retention")


def fmt_table(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(d[c]):
            if any(k in c.lower() for k in PCT_KEYS) and d[c].abs().max() <= 5:
                d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
            else:
                d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}")
    ACR = {"Wape": "WAPE", "Mape": "MAPE", "Mae": "MAE", "Rmse": "RMSE", "Aov": "AOV", "Rfm": "RFM", "Id": "ID"}
    d.columns = [" ".join(ACR.get(w, w) for w in str(c).replace("_", " ").title().split()) for c in d.columns]
    return d.to_html(index=False, border=0, classes="tbl", escape=True)


def apply_filters(df: pd.DataFrame, args) -> tuple[pd.DataFrame, dict]:
    sel = {k: args.get(k, "All") for k in ("year", "market", "category", "segment")}
    out = df
    if sel["year"] != "All":
        out = out[out.year == int(sel["year"])]
    for k in ("market", "category", "segment"):
        if sel[k] != "All":
            out = out[out[k] == sel[k]]
    return out, sel


def kpi_cards(k):
    f = lambda v, p="$": "n/a" if pd.isna(v) else f"{p}{v:,.0f}"
    return {"type": "kpis", "items": [("Revenue", f(k["revenue"])), ("Reported profit", f(k["profit"])),
            ("Profit margin", "n/a" if pd.isna(k["margin"]) else f"{k['margin']:.1%}"), ("Orders", f"{k['orders']:,}"),
            ("Avg order value", f(k["aov"])), ("Customers", f"{k['customers']:,}"),
            ("Avg discount", "n/a" if pd.isna(k["avg_discount"]) else f"{k['avg_discount']:.1%}")]}


def note(text, kind="info"):
    return {"type": "note", "text": text, "kind": kind}


# ------------------------------- pages -------------------------------
def page_overview(df, S):
    m = A.monthly(df)
    fig = go.Figure([go.Scatter(x=m.month, y=m.revenue, name="Revenue", mode="lines"),
                     go.Scatter(x=m.month, y=m.profit, name="Reported profit", mode="lines")])
    cat = A.by_dimension(df, "category")
    mk = A.by_dimension(df, "market")
    return [kpi_cards(A.kpis(df)),
            chart(fig, "Monthly revenue and reported profit", "Profit is the source's reported field, not audited net profit.", True),
            chart(go.Figure(go.Bar(x=cat.category, y=cat.revenue, marker_color=PALETTE[1])), "Revenue by category"),
            chart(go.Figure(go.Bar(x=mk.market, y=mk.margin, marker_color=PALETTE[0])).update_yaxes(tickformat=".0%"), "Reported margin by market"),
            table(m.sort_values("month", ascending=False), "monthly", "Monthly KPI table", n=12)]


def page_sales(df, S):
    m = A.monthly(df)
    s = A.seasonality(df)
    fig = go.Figure([go.Scatter(x=s.m, y=s[c], name=str(c), mode="lines+markers") for c in s.columns if c != "m"])
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Index (year's monthly mean = 1)")
    yoy = m.dropna(subset=["revenue_yoy"])
    seg = A.by_dimension(df, "segment")
    sm = A.by_dimension(df, "ship_mode")
    return [kpi_cards(A.kpis(df)),
            chart(go.Figure(go.Bar(x=yoy.month, y=yoy.revenue_yoy, marker_color=PALETTE[1])).update_yaxes(tickformat=".0%"), "Revenue growth vs. same month last year",
                  "Needs at least 12 months of history in the current filter."),
            chart(fig, "Seasonality by year (revenue index)", "Values above 1 are stronger-than-average months."),
            chart(go.Figure(go.Scatter(x=m.month, y=m.aov, mode="lines")), "Average order value"),
            chart(go.Figure(go.Bar(x=seg.segment, y=seg.revenue, marker_color=PALETTE[0])), "Revenue by customer segment"),
            table(sm, "shipmode", "Performance by ship mode"), table(seg, "segment", "Performance by customer segment")]


def page_products(df, S):
    sub = A.by_dimension(df, ["category", "sub_category"]).sort_values("profit")
    fig = go.Figure(go.Bar(y=sub.sub_category, x=sub.profit, orientation="h",
                           marker_color=np.where(sub.profit >= 0, PALETTE[3], PALETTE[4])))
    fig.update_layout(height=520)
    blocks = [kpi_cards(A.kpis(df)), chart(fig, "Reported profit by sub-category", wide=True)]
    if df.year.nunique() >= 2:
        opp = C.opportunity_matrix(df)
        sc = go.Figure(go.Scatter(x=opp.growth, y=opp.margin, mode="markers+text", text=opp.sub_category, textposition="top center",
                                  marker=dict(size=np.sqrt(opp.revenue) / 12 + 6, color=PALETTE[1], opacity=0.7)))
        sc.update_xaxes(title="Revenue growth, latest year in filter", tickformat=".0%")
        sc.update_yaxes(title="Reported margin", tickformat=".0%")
        blocks.append(chart(sc, "Product opportunity matrix", "Bubble size = revenue. Dashed lines = medians.", True))
        blocks.append(table(opp.sort_values("revenue", ascending=False), "opportunity", "Opportunity quadrants"))
    else:
        blocks.append(note("Select 'All' years (or more than one year) to see growth-based opportunity analysis."))
    blocks += [table(A.top_bottom_products(df), "products", "Top 10 and bottom 10 products by profit", n=20),
               table(C.weak_profit_products(df), "weak_products", "High-revenue products with under 2% margin", n=15)]
    return blocks


def page_regions(df, S):
    mk = A.by_dimension(df, ["market", "year"]).sort_values(["market", "year"])
    fig = go.Figure([go.Bar(x=g.market, y=g.revenue, name=str(y)) for y, g in mk.groupby("year")])
    fig.update_layout(barmode="group")
    ctry = A.by_dimension(df, "country").head(15)
    return [kpi_cards(A.kpis(df)), chart(fig, "Revenue by market and year", wide=True),
            chart(go.Figure(go.Bar(x=ctry.country, y=ctry.revenue, marker_color=PALETTE[1])), "Top 15 countries by revenue"),
            chart(go.Figure(go.Bar(x=ctry.country, y=ctry.margin, marker_color=PALETTE[0])).update_yaxes(tickformat=".0%"), "Reported margin, top 15 countries"),
            table(A.by_dimension(df, "market"), "market", "Market summary"),
            table(A.by_dimension(df, "country"), "country", "Country summary", n=20)]


def page_customers(df, S):
    r = A.rfm(df)
    seg = A.segment_profile(r)
    blocks = [kpi_cards(A.kpis(df)),
              chart(go.Figure(go.Bar(x=seg.segment, y=seg.revenue_share, marker_color=PALETTE[1])).update_yaxes(tickformat=".0%"), "Revenue share by RFM segment"),
              chart(go.Figure(go.Bar(x=seg.segment, y=seg.margin, marker_color=PALETTE[0])).update_yaxes(tickformat=".0%"), "Reported margin by RFM segment")]
    coh = A.cohort_retention(df)
    z = coh.drop(columns=["cohort", "cohort_size"])
    hm = go.Figure(go.Heatmap(z=z.values, x=[f"Q+{c}" for c in z.columns], y=coh.cohort, colorscale="Blues", zmin=0, zmax=1,
                              hovertemplate="Cohort %{y}<br>%{x}: %{z:.0%}<extra></extra>"))
    hm.update_layout(height=460)
    blocks.append(chart(hm, "Quarterly cohort retention", "Share of each first-purchase cohort that buys again N quarters later.", True))
    p = A.pareto(df)
    blocks.append(note(f"The top 20% of customers in this view generate {p['top20pct_revenue_share']:.0%} of revenue. "
                       "Customer value below is HISTORICAL revenue/profit to date, not a predicted lifetime value."))
    blocks += [table(seg, "segments", "RFM segment profile"), table(A.customer_value(df), "customer_value", "Highest-value customers (historical)")]
    ch = S.r["churn"]
    blocks.append(note(f"Churn model (time-split): AUC {S.r['metrics']['churn']['auc_logistic']:.2f} vs recency-only baseline "
                       f"{S.r['metrics']['churn']['auc_recency_baseline']:.2f}. The list below is scored as of 2014-06-30 for all customers (not filtered). "
                       "'Churn' = no purchase in the following 180 days.", "info"))
    blocks.append(table(ch[["customer_id", "recency", "frequency", "profit", "churn_probability"]], "churn_risk",
                        "Customers at highest risk of lapsing (as of 2014-06-30)", filters=False))
    return blocks


def page_forecasting(df, S):
    r, m = S.r, S.r["metrics"]["forecast"]
    best = m["best_model"]
    w = r["weekly"].tail(110)
    test = r["preds"][(r["preds"].model_name == best) & (r["preds"].stage == "test")]
    naive = r["preds"][(r["preds"].model_name == "Seasonal naive") & (r["preds"].stage == "test")]
    fut = r["future"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w.week_ending, y=w.revenue, mode="lines", name="Actual weekly revenue", line=dict(color=PALETTE[0])))
    fig.add_trace(go.Scatter(x=test.forecast_date, y=test.predicted, mode="lines", name=f"{best}: held-out test forecast", line=dict(color=PALETTE[2], dash="dot")))
    fig.add_trace(go.Scatter(x=naive.forecast_date, y=naive.predicted, mode="lines", name="Seasonal naive (baseline)", line=dict(color="#999", dash="dot")))
    fig.add_trace(go.Scatter(x=fut.week_ending, y=fut.revenue_high, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut.week_ending, y=fut.revenue_low, mode="lines", fill="tonexty", fillcolor="rgba(46,117,182,0.18)", line=dict(width=0), name="Approx. 80% interval"))
    fig.add_trace(go.Scatter(x=fut.week_ending, y=fut.revenue_base, mode="lines+markers", name="Forecast: next 13 weeks", line=dict(color=PALETTE[1])))
    test_wape = m["test"][best]["WAPE"]
    sel = r["selection"].copy()
    tst = r["fc_results"][r["fc_results"].stage == "test"][["model", "WAPE", "MAE", "RMSE"]].rename(columns={"WAPE": "test_WAPE", "MAE": "test_MAE", "RMSE": "test_RMSE"})
    comp = sel.merge(tst, on="model").rename(columns={"WAPE": "validation_WAPE", "MAE": "validation_MAE", "RMSE": "validation_RMSE"})
    comp = comp[["model", "validation_WAPE", "validation_MAE", "test_WAPE", "test_MAE", "test_RMSE"]]
    cat = r["cat_eval"]
    return [note(f"Selected model: {best}, chosen on three rolling-origin validation folds only. Held-out test WAPE: {test_wape:.1%}. "
                 f"The 13-week base case is {m['plausibility']['ratio']:.2f}x the same weeks last year (trailing 52-week growth: {m['plausibility']['trailing_52w_growth']:.0%}). "
                 "Forecasts are at total-revenue level; the interval is empirical and approximate."),
            chart(fig, "Weekly revenue: actual, held-out test forecast and next-13-week forecast", wide=True),
            table(comp, "forecast_comparison", "Model comparison", n=10, filters=False),
            table(cat, "category_forecast", "Accuracy by category (same protocol)", n=20, filters=False),
            table(fut, "forecast_next13", "Next-13-week forecast", n=13, filters=False)]


def page_discounts(df, S):
    b = C.discount_bands(df)
    fig = go.Figure(go.Bar(x=b.discount_band.astype(str), y=b.margin, marker_color=np.where(b.margin >= 0, PALETTE[3], PALETTE[4])))
    fig.update_yaxes(tickformat=".0%")
    fig2 = go.Figure(go.Bar(x=b.discount_band.astype(str), y=b.loss_line_share, marker_color=PALETTE[4]))
    fig2.update_yaxes(tickformat=".0%")
    return [kpi_cards(A.kpis(df)),
            note("These are descriptive associations. Discounts are not randomly assigned, so nothing here shows that a discount caused a margin change.", "warn"),
            chart(fig, "Reported margin by discount band"), chart(fig2, "Share of order lines with negative profit"),
            table(b, "discount_bands", "Discount band summary"), table(C.discount_vs_nodiscount(df), "discount_vs_full", "Discounted vs. full-price lines"),
            table(C.discount_by_subcategory(df), "discount_subcat", "Discount behaviour by sub-category", n=20)]


def page_planning(df, S):
    p = S.r["plan"]
    fig = go.Figure([go.Scatter(x=p.week_ending, y=p[f"units_{s}"], mode="lines", name=s.title()) for s in ("low", "base", "high")])
    fig.update_yaxes(title="Units per week")
    return [note("Scenario view only. The source contains no inventory, lead-time or supplier data, so reorder points and safety stock are not computed. "
                 f"Units are inferred from the trailing-52-week units-per-revenue-dollar ratio ({p.units_per_revenue_dollar.iloc[0]:.5f}).", "warn"),
            chart(fig, "Forecast weekly unit demand, low / base / high", wide=True),
            table(C.horizon_summary(p), "planning_horizon", "30 / 60 / 90-day demand scenarios", n=5, filters=False),
            table(p[["week_ending", "revenue_low", "revenue_base", "revenue_high", "units_base", "cum_units_base"]], "planning_weekly", "Weekly plan", n=13, filters=False)]


DOWNLOADS = {"excel": (config.EXCEL_DIR / "sales_intelligence_report.xlsx", "Excel workbook (17 tabs)"),
             "html": (config.HTML_DIR / "management_report.html", "Management report (HTML; print to PDF)"),
             "forecast": (config.RESULTS_DIR / "forecast_next13.csv", "Next-13-week forecast (CSV)"),
             "evaluation": (config.RESULTS_DIR / "forecast_evaluation.csv", "Forecast evaluation (CSV)"),
             "churn": (config.RESULTS_DIR / "churn_scores_2014H2.csv", "Churn risk scores (CSV)"),
             "metrics": (config.RESULTS_DIR / "metrics.json", "Run metrics (JSON)")}


def page_reports(df, S):
    return [{"type": "downloads", "items": [(k, d, p.exists()) for k, (p, d) in DOWNLOADS.items()]}]


PAGES = {"overview": page_overview, "sales": page_sales, "products": page_products, "regions": page_regions,
         "customers": page_customers, "forecasting": page_forecasting, "discounts": page_discounts,
         "planning": page_planning, "reports": page_reports}


# ------------------------------- routes -------------------------------
def _store() -> DataStore:
    return current_app.config["STORE"]


@bp.route("/")
@bp.route("/<page>")
def page(page="overview"):
    if page not in PAGES:
        abort(404)
    S = _store()
    df, sel = apply_filters(S.df, request.args) if page in FILTERABLE else (S.df, {})
    title = dict(NAV)[page]
    ctx = dict(nav=NAV, page=page, title=title, sel=sel, filterable=page in FILTERABLE,
               opts={"year": sorted(S.df.year.unique()), "market": sorted(S.df.market.unique()),
                     "category": sorted(S.df.category.unique()), "segment": sorted(S.df.segment.unique())},
               query=request.query_string.decode())
    if page in FILTERABLE and df.empty:
        ctx["blocks"] = [note("No data matches the selected filters. Adjust or reset the filters.", "warn")]
    else:
        ctx["blocks"] = PAGES[page](df, S)
    return render_template("page.html", **ctx)


TABLES = {"monthly": lambda d, S: A.monthly(d), "shipmode": lambda d, S: A.by_dimension(d, "ship_mode"),
          "segment": lambda d, S: A.by_dimension(d, "segment"), "products": lambda d, S: A.top_bottom_products(d, 25),
          "weak_products": lambda d, S: C.weak_profit_products(d), "market": lambda d, S: A.by_dimension(d, "market"),
          "country": lambda d, S: A.by_dimension(d, "country"), "segments": lambda d, S: A.segment_profile(A.rfm(d)),
          "customer_value": lambda d, S: A.customer_value(d), "discount_bands": lambda d, S: C.discount_bands(d),
          "discount_vs_full": lambda d, S: C.discount_vs_nodiscount(d), "discount_subcat": lambda d, S: C.discount_by_subcategory(d),
          "opportunity": lambda d, S: C.opportunity_matrix(d),
          "churn_risk": lambda d, S: S.r["churn"], "forecast_comparison": lambda d, S: S.r["selection"],
          "category_forecast": lambda d, S: S.r["cat_eval"], "forecast_next13": lambda d, S: S.r["future"],
          "planning_horizon": lambda d, S: C.horizon_summary(S.r["plan"]), "planning_weekly": lambda d, S: S.r["plan"]}


@bp.route("/export/<name>.csv")
def export_table(name):
    if name not in TABLES:
        abort(404)
    S = _store()
    df, _ = apply_filters(S.df, request.args)
    out = TABLES[name](df, S) if not df.empty else pd.DataFrame()
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={name}.csv"})


@bp.route("/download/<key>")
def download(key):
    if key not in DOWNLOADS or not DOWNLOADS[key][0].exists():
        abort(404)
    return send_file(DOWNLOADS[key][0], as_attachment=True)


@bp.route("/health")
def health():
    S = _store()
    return {"status": "ok", "rows": len(S.df), "revenue": float(S.df.sales.sum())}
