import pandas as pd

from salesintel import analytics as A


def test_kpi_definitions(df):
    k = A.kpis(df)
    assert abs(k["aov"] - k["revenue"] / k["orders"]) < 1e-9
    assert abs(k["margin"] - k["profit"] / k["revenue"]) < 1e-12


def test_rfm_reconciles_with_transactions(df):
    r = A.rfm(df).set_index("customer_id")
    snap = df.order_date.max() + pd.Timedelta(days=1)
    cid = df.customer_id.iloc[1234]
    d = df[df.customer_id == cid]
    assert r.loc[cid, "frequency"] == d.order_key.nunique()
    assert abs(r.loc[cid, "monetary"] - d.sales.sum()) < 1e-6
    assert r.loc[cid, "recency"] == (snap - d.order_date.max()).days


def test_rfm_scores_and_coverage(df):
    r = A.rfm(df)
    assert r[["R", "F", "M"]].stack().between(1, 5).all()
    assert len(r) == df.customer_id.nunique() and r.segment.notna().all()
    assert abs(A.segment_profile(r).revenue.sum() - df.sales.sum()) < 1e-3


def test_cohort_first_period_is_full(df):
    c = A.cohort_retention(df)
    assert (c[0] == 1.0).all()


def test_pareto_in_range(df):
    assert 0.2 < A.pareto(df)["top20pct_revenue_share"] <= 1


def test_filtered_kpis_sum_to_total(df):
    parts = sum(A.kpis(df[df.year == y])["revenue"] for y in df.year.unique())
    assert abs(parts - A.kpis(df)["revenue"]) < 1e-3


def test_churn_features_use_only_past_data(df):
    cutoff = pd.Timestamp("2014-06-30")
    full, _ = A._churn_frame(df, cutoff)
    trunc, _ = A._churn_frame(df[df.order_date <= cutoff], cutoff)
    pd.testing.assert_frame_equal(full.sort_index(), trunc.sort_index())


def test_churn_model_beats_chance(pipe):
    ch = pipe["churn"]
    assert ch["auc_logistic"] > 0.6
