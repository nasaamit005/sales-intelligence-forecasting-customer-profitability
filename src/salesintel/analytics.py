"""Descriptive and diagnostic analytics. All functions take a (possibly filtered) transactions frame."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def safe_div(a, b):
    return a / b if b else np.nan


def kpis(df: pd.DataFrame) -> dict:
    rev, prof = df.sales.sum(), df.profit.sum()
    orders = df.order_key.nunique()
    return {"revenue": rev, "profit": prof, "margin": safe_div(prof, rev), "orders": orders,
            "aov": safe_div(rev, orders), "units": int(df.quantity.sum()),
            "customers": df.customer_id.nunique(), "avg_discount": df.discount.mean() if len(df) else np.nan,
            "shipping_cost_ratio": safe_div(df.shipping_cost.sum(), rev)}


def _agg(g):
    return g.agg(revenue=("sales", "sum"), profit=("profit", "sum"), units=("quantity", "sum"),
                 orders=("order_key", "nunique"), avg_discount=("discount", "mean"),
                 shipping_cost=("shipping_cost", "sum"))


def with_margin(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["margin"] = t["profit"] / t["revenue"]
    return t


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    t = with_margin(_agg(df.groupby("month"))).reset_index()
    t["aov"] = t["revenue"] / t["orders"]
    t["revenue_yoy"] = t["revenue"].pct_change(12)
    return t


def by_dimension(df: pd.DataFrame, cols) -> pd.DataFrame:
    return with_margin(_agg(df.groupby(cols))).reset_index().sort_values("revenue", ascending=False)


def yearly_growth(df: pd.DataFrame, col: str) -> pd.DataFrame:
    p = df.pivot_table(index=col, columns="year", values="sales", aggfunc="sum")
    p.columns = [f"revenue_{c}" for c in p.columns]
    last, prev = p.columns[-1], p.columns[-2] if p.shape[1] > 1 else None
    if prev:
        p["growth_last_year"] = p[last] / p[prev] - 1
    return p.reset_index()


def seasonality(df: pd.DataFrame) -> pd.DataFrame:
    t = df.assign(m=df.order_date.dt.month).pivot_table(index="m", columns="year", values="sales", aggfunc="sum")
    return (t / t.mean()).reset_index()      # index of each month vs that year's monthly mean


def top_bottom_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    t = with_margin(_agg(df.groupby(["product_name", "category", "sub_category"]))).reset_index()
    return t.sort_values("profit", ascending=False).pipe(lambda x: pd.concat([x.head(n), x.tail(n)]))


# ----------------------------- customers -----------------------------
SEGMENT_RULES = [("Champions", lambda r: r.R >= 4 and r.F >= 4), ("Loyal", lambda r: r.R >= 3 and r.F >= 4),
                 ("Potential Loyalists", lambda r: r.R >= 4 and r.F >= 2), ("New Customers", lambda r: r.R >= 4),
                 ("At Risk", lambda r: r.R <= 2 and r.F >= 3), ("Hibernating", lambda r: r.R <= 2)]


def rfm(df: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    snapshot = snapshot or (df.order_date.max() + pd.Timedelta(days=1))
    g = df.groupby("customer_id")
    t = pd.DataFrame({"recency": (snapshot - g.order_date.max()).dt.days,
                      "frequency": g.order_key.nunique(), "monetary": g.sales.sum(),
                      "profit": g.profit.sum()})
    if len(t) < 5:
        return t.assign(R=3, F=3, M=3, segment="All")
    t["R"] = pd.qcut(t.recency.rank(method="first", ascending=False), 5, labels=False) + 1
    t["F"] = pd.qcut(t.frequency.rank(method="first"), 5, labels=False) + 1
    t["M"] = pd.qcut(t.monetary.rank(method="first"), 5, labels=False) + 1
    t["segment"] = t.apply(lambda r: next((n for n, f in SEGMENT_RULES if f(r)), "Others"), axis=1)
    return t.reset_index()


def segment_profile(r: pd.DataFrame) -> pd.DataFrame:
    p = r.groupby("segment").agg(customers=("customer_id", "count"), avg_recency=("recency", "mean"),
                                 avg_frequency=("frequency", "mean"), revenue=("monetary", "sum"),
                                 profit=("profit", "sum")).reset_index()
    p["revenue_share"] = p["revenue"] / p["revenue"].sum()
    p["margin"] = p["profit"] / p["revenue"]
    return p.sort_values("revenue", ascending=False)


def cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each first-purchase-quarter cohort that buys again N quarters later."""
    d = df[["customer_id", "order_date"]].drop_duplicates().copy()
    d["q"] = d.order_date.dt.to_period("Q")
    first = d.groupby("customer_id")["q"].min().rename("cohort")
    d = d.merge(first, on="customer_id")
    d["age"] = (d["q"] - d["cohort"]).apply(lambda x: x.n)
    size = d.groupby("cohort")["customer_id"].nunique()
    act = d.groupby(["cohort", "age"])["customer_id"].nunique().unstack(fill_value=0)
    ret = act.div(size, axis=0)
    # blank out quarters that lie beyond the observed data window
    last_q = d["q"].max()
    for c in ret.index:
        max_age = (last_q - c).n
        ret.loc[c, ret.columns > max_age] = np.nan
    ret.index = ret.index.astype(str)
    ret.insert(0, "cohort_size", size.values)
    return ret.reset_index()


def customer_value(df: pd.DataFrame) -> pd.DataFrame:
    """HISTORICAL customer value (revenue and reported profit to date). Not a forward-looking CLV."""
    g = df.groupby(["customer_id", "customer_name", "segment"])
    t = g.agg(first_order=("order_date", "min"), last_order=("order_date", "max"),
              orders=("order_key", "nunique"), revenue=("sales", "sum"), profit=("profit", "sum")).reset_index()
    t["tenure_days"] = (t.last_order - t.first_order).dt.days
    t["aov"] = t.revenue / t.orders
    t["historical_profit_per_order"] = t.profit / t.orders
    t["value_tier"] = pd.qcut(t.profit.rank(method="first"), 4, labels=["Q4 low", "Q3", "Q2", "Q1 high"]).astype(str)
    return t.sort_values("profit", ascending=False)


def pareto(df: pd.DataFrame) -> dict:
    c = df.groupby("customer_id").sales.sum().sort_values(ascending=False)
    share = c.cumsum() / c.sum()
    top20 = share.iloc[int(np.ceil(len(c) * 0.2)) - 1] if len(c) else np.nan
    return {"customers": len(c), "top20pct_revenue_share": float(top20)}


# ----------------------------- churn model -----------------------------
def _churn_frame(df: pd.DataFrame, cutoff: pd.Timestamp, horizon_days: int = 180):
    hist = df[df.order_date <= cutoff]
    fut = df[(df.order_date > cutoff) & (df.order_date <= cutoff + pd.Timedelta(days=horizon_days))]
    g = hist.groupby("customer_id")
    X = pd.DataFrame({"recency": (cutoff - g.order_date.max()).dt.days, "frequency": g.order_key.nunique(),
                      "monetary": g.sales.sum(), "profit": g.profit.sum(),
                      "tenure": (cutoff - g.order_date.min()).dt.days, "avg_discount": g.discount.mean(),
                      "orders_last_180d": hist[hist.order_date > cutoff - pd.Timedelta(days=180)]
                      .groupby("customer_id").order_key.nunique()}).fillna(0)
    X["log_monetary"] = np.log1p(X.monetary.clip(lower=0))
    y = (~X.index.isin(fut.customer_id.unique())).astype(int)       # 1 = no purchase in next 180 days
    return X.drop(columns=["monetary"]), pd.Series(y, index=X.index, name="churned")


def churn_model(df: pd.DataFrame) -> dict:
    """Time-split churn model: train on the 2013-06-30 cutoff, test on the 2014-06-30 cutoff.

    Label = no purchase in the following 180 days. Test labels are strictly in the future of
    every training feature and label window.
    """
    Xtr, ytr = _churn_frame(df, pd.Timestamp("2013-06-30"))
    Xte, yte = _churn_frame(df, pd.Timestamp("2014-06-30"))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE))
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    base = Xte["recency"].rank(pct=True).values        # naive baseline: longer inactivity = higher risk
    scored = Xte.assign(churn_probability=p, churned_actual=yte.values).reset_index()
    coef = pd.Series(model[-1].coef_[0], index=Xtr.columns).sort_values()
    return {"auc_logistic": float(roc_auc_score(yte, p)), "auc_recency_baseline": float(roc_auc_score(yte, base)),
            "train_customers": int(len(Xtr)), "test_customers": int(len(Xte)),
            "train_churn_rate": float(ytr.mean()), "test_churn_rate": float(yte.mean()),
            "scored": scored.sort_values("churn_probability", ascending=False),
            "coefficients": coef.rename("standardised_coefficient").reset_index().rename(columns={"index": "feature"})}
