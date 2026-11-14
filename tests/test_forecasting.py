import numpy as np
import pandas as pd

from salesintel import config
from salesintel import forecasting as F


def test_metric_values():
    m = F.metrics([100, 200], [110, 180])
    assert m["MAE"] == 15 and abs(m["WAPE"] - 30 / 300) < 1e-12
    assert abs(m["RMSE"] - np.sqrt((100 + 400) / 2)) < 1e-9
    assert abs(m["Bias"] - (-10 + 20) / 300) < 1e-12


def test_weekly_series_is_complete_and_contiguous(df):
    y = F.weekly_series(df)
    assert (y.index.weekday == 6).all()
    assert (np.diff(y.index.values).astype("timedelta64[D]") == np.timedelta64(7, "D")).all()
    inside = df[(df.order_date > y.index[0] - pd.Timedelta(days=7)) & (df.order_date <= y.index[-1])]
    assert abs(inside.sales.sum() - y.sum()) < 1e-6


def test_lag_features_never_see_the_future():
    idx = pd.date_range("2012-01-01", periods=120, freq="W-SUN")
    y = pd.Series(np.arange(120, dtype=float), index=idx)
    base = F.make_features(y)
    y2 = y.copy()
    y2.iloc[80:] += 1e6                                         # change everything from t=80 onward
    changed = F.make_features(y2)
    pd.testing.assert_frame_equal(base.iloc[:81], changed.iloc[:81])   # rows <= 80 unaffected


def test_split_is_chronological_and_disjoint():
    sp = F.split_points(208)
    assert sp["test_start"] == 208 - config.FORECAST_HORIZON_WEEKS
    assert sp["val_origins"] == sorted(sp["val_origins"])
    assert sp["val_origins"][-1] + config.FORECAST_HORIZON_WEEKS <= sp["test_start"]


def test_seasonal_naive_repeats_last_year():
    idx = pd.date_range("2012-01-01", periods=110, freq="W-SUN")
    y = pd.Series(np.arange(110, dtype=float), index=idx)
    fc = F.fc_snaive(y, 4)
    assert fc.tolist() == y.iloc[-52:-48].tolist()


def test_selection_ignores_test_block(df):
    y = F.weekly_series(df)
    ev1 = F.evaluate(y)
    y2 = y.copy()
    y2.iloc[F.split_points(len(y))["test_start"]:] *= 10        # corrupt the held-out test block
    ev2 = F.evaluate(y2)
    pd.testing.assert_frame_equal(ev1["selection"], ev2["selection"])
    assert ev1["best_model"] == ev2["best_model"]


def test_best_model_is_lowest_validation_wape(pipe):
    ev = pipe["evaluation"]
    assert ev["best_model"] == ev["selection"].sort_values("WAPE").iloc[0]["model"]
    assert set(ev["selection"].model) == set(F.MODELS)          # baselines are included


def test_future_forecast_shape_and_ordering(pipe):
    f = pipe["forecast"]
    assert len(f) == config.FORECAST_HORIZON_WEEKS
    assert (f.revenue_low <= f.revenue_base).all() and (f.revenue_base <= f.revenue_high).all()
    assert f.week_ending.is_monotonic_increasing


def test_forecast_is_reproducible(df):
    y = F.weekly_series(df)
    a, b = F.fc_xgb(y.iloc[:150], 6), F.fc_xgb(y.iloc[:150], 6)
    pd.testing.assert_series_equal(a, b)


def test_yoy_features_never_see_the_future():
    idx = pd.date_range("2012-01-01", periods=140, freq="W-SUN")
    y = pd.Series(np.linspace(100, 300, 140), index=idx)
    base = F._yoy_features(y)
    y2 = y.copy()
    y2.iloc[100:] *= 50
    pd.testing.assert_frame_equal(base.iloc[:101], F._yoy_features(y2).iloc[:101])


def test_final_forecast_is_plausible_vs_last_year(pipe, df):
    """Regression guard: an earlier level-based model overstated Q1 demand by ~70% vs last year."""
    pl = pipe["metrics"]["forecast"]["plausibility"]
    assert 0.8 <= pl["ratio"] <= 1.6
    assert abs(pl["ratio"] - 1 - pl["trailing_52w_growth"]) < 0.25   # growth of the same order as recent trend


def test_growth_baseline_scales_last_year():
    idx = pd.date_range("2012-01-01", periods=110, freq="W-SUN")
    y = pd.Series(np.r_[np.full(55, 100.0), np.full(55, 200.0)], index=idx)
    g = y.iloc[-52:].sum() / y.iloc[-104:-52].sum()
    assert abs(F.fc_snaive_growth(y, 3).iloc[0] - y.iloc[-52] * g) < 1e-9
