"""Commercial analytics: discounts, product opportunity matrix and forecast-based demand planning."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def discount_bands(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("discount_band")
    t = g.agg(lines=("sales", "size"), revenue=("sales", "sum"), profit=("profit", "sum"),
              avg_quantity=("quantity", "mean"), loss_line_share=("profit", lambda s: (s < 0).mean())).reset_index()
    t["margin"] = t["profit"] / t["revenue"]
    t["discount_band"] = pd.Categorical(t["discount_band"], config.DISCOUNT_LABELS, ordered=True)
    return t.sort_values("discount_band")


def discount_vs_nodiscount(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive comparison only. Discounts are not randomly assigned, so this is NOT a causal estimate."""
    flag = np.where(df.discount > 0, "Discounted", "Full price")
    g = df.groupby(flag).agg(lines=("sales", "size"), avg_quantity=("quantity", "mean"),
                             avg_line_sales=("sales", "mean"), avg_profit=("profit", "mean"),
                             margin=("profit", lambda s: s.sum()), revenue=("sales", "sum"))
    g["margin"] = g["margin"] / g["revenue"]
    return g.reset_index().rename(columns={"index": "price_type"})


def discount_by_subcategory(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.groupby("sub_category").apply(
        lambda d: pd.Series({"revenue": d.sales.sum(), "profit": d.profit.sum(), "avg_discount": d.discount.mean(),
                             "corr_discount_margin": d.discount.corr(d.profit_margin)}), include_groups=False)
    sub["margin"] = sub["profit"] / sub["revenue"]
    return sub.reset_index().sort_values("avg_discount", ascending=False)


def weak_profit_products(df: pd.DataFrame, min_revenue: float = 5000) -> pd.DataFrame:
    t = df.groupby(["product_name", "category", "sub_category"]).agg(
        revenue=("sales", "sum"), profit=("profit", "sum"), avg_discount=("discount", "mean")).reset_index()
    t["margin"] = t.profit / t.revenue
    return t[(t.revenue >= min_revenue) & (t.margin < 0.02)].sort_values("revenue", ascending=False)


def opportunity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Sub-category growth (last full year vs prior) against margin, with a quadrant label."""
    y1, y0 = df.year.max(), df.year.max() - 1
    rev = df.pivot_table(index="sub_category", columns="year", values="sales", aggfunc="sum")
    t = pd.DataFrame({"revenue": rev[y1], "growth": rev[y1] / rev[y0] - 1})
    t["margin"] = df[df.year == y1].groupby("sub_category").apply(lambda d: d.profit.sum() / d.sales.sum(), include_groups=False)
    mg, gm = t.margin.median(), t.growth.median()
    t["quadrant"] = np.select([(t.growth >= gm) & (t.margin >= mg), (t.growth >= gm) & (t.margin < mg),
                               (t.growth < gm) & (t.margin >= mg)],
                              ["Invest (growing, profitable)", "Fix margin (growing, thin margin)",
                               "Harvest (slower growth, profitable)"], "Review (slower growth, thin margin)")
    return t.reset_index()


def demand_planning(forecast: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Translate the weekly revenue forecast into unit-demand SCENARIOS.

    Units are inferred from the trailing-52-week units-per-revenue-dollar ratio. No inventory,
    lead-time or supplier data exists in the source, so reorder points / safety stock are
    deliberately NOT computed.
    """
    recent = df[df.order_date > df.order_date.max() - pd.Timedelta(weeks=52)]
    upd = recent.quantity.sum() / recent.sales.sum()
    f = forecast.copy()
    for s in ("low", "base", "high"):
        f[f"units_{s}"] = f[f"revenue_{s}"] * upd
    f["horizon_week"] = np.arange(1, len(f) + 1)
    f["cum_revenue_base"] = f["revenue_base"].cumsum()
    f["cum_units_base"] = f["units_base"].cumsum()
    f["units_per_revenue_dollar"] = upd
    return f


def horizon_summary(plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for days, wk in ((30, 4), (60, 9), (90, 13)):
        p = plan.head(wk)
        rows.append({"horizon": f"~{days} days ({wk} weeks)", "revenue_low": p.revenue_low.sum(),
                     "revenue_base": p.revenue_base.sum(), "revenue_high": p.revenue_high.sum(),
                     "units_low": p.units_low.sum(), "units_base": p.units_base.sum(), "units_high": p.units_high.sum()})
    return pd.DataFrame(rows)
