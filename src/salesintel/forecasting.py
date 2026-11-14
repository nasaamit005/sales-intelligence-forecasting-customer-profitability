"""Weekly sales forecasting with time-aware validation.

Design
------
* Series: weekly (Mon-Sun) revenue; partial first/last weeks are dropped.
* Models: naive, seasonal naive (52w), seasonal naive + growth, 4-week moving average, ARIMA with
  Fourier seasonality, XGBoost on levels, and XGBoost on the year-over-year ratio (recursive multi-step).
* Selection: expanding-window, rolling-origin folds on data BEFORE the final test block.
  The last 13 weeks are a held-out test set and are never used to choose a model.
* Final forecast: best model (by validation WAPE) is refit on all data for the next 13 weeks;
  an empirical interval is calibrated from validation-fold forecast-error ratios.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

from . import config

H = config.FORECAST_HORIZON_WEEKS
SP = config.SEASONAL_PERIOD
MODELS = ["Naive", "Seasonal naive", "Seasonal naive + growth", "Moving average (4w)", "ARIMA + Fourier",
          "XGBoost (levels)", "XGBoost (YoY ratio)"]


# ------------------------------- series -------------------------------
def weekly_series(df: pd.DataFrame, value: str = "sales") -> pd.Series:
    s = df.set_index("order_date")[value].resample("W-SUN").sum()
    start = df.order_date.min()
    end = df.order_date.max()
    # drop partial first / last week (week-ending Sunday label)
    if start.weekday() != 0:
        s = s.iloc[1:]
    if end.weekday() != 6:
        s = s.iloc[:-1]
    return s.astype(float)


# ------------------------------- metrics -------------------------------
def metrics(actual, pred) -> dict:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    e = a - p
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "WAPE": float(np.abs(e).sum() / np.abs(a).sum()),
            "MAPE": float(np.mean(np.abs(e / a))), "Bias": float(e.sum() / a.sum())}


# ------------------------------- features -------------------------------
LAGS = (1, 2, 3, 4, 8, 13, 52)


def make_features(y: pd.Series, origin_len: int | None = None) -> pd.DataFrame:
    """Feature row for each week t uses only values at weeks < t (strict shift)."""
    f = pd.DataFrame(index=y.index)
    for l in LAGS:
        f[f"lag_{l}"] = y.shift(l)
    sh = y.shift(1)
    f["roll_mean_4"] = sh.rolling(4).mean()
    f["roll_mean_13"] = sh.rolling(13).mean()
    f["roll_std_4"] = sh.rolling(4).std()
    f["week_of_year"] = y.index.isocalendar().week.astype(int).values
    f["month"] = y.index.month
    # NOTE: no raw time-index feature. Trees cannot extrapolate it, and it made level forecasts snap back to
    # the latest (peak-season) leaf, overstating Q1 demand by ~50%.
    return f


def _features_for_next(history: pd.Series, next_date: pd.Timestamp) -> pd.DataFrame:
    """Build one feature row for next_date from history only (used for recursive forecasting)."""
    ext = pd.concat([history, pd.Series([np.nan], index=[next_date])])
    return make_features(ext).iloc[[-1]]


# ------------------------------- forecasters -------------------------------
def _future_index(train: pd.Series, h: int) -> pd.DatetimeIndex:
    return pd.date_range(train.index[-1] + pd.Timedelta(weeks=1), periods=h, freq="W-SUN")


def fc_naive(train, h):
    return pd.Series(train.iloc[-1], index=_future_index(train, h))


def fc_snaive(train, h):
    vals = [train.iloc[-SP + (i % SP)] for i in range(h)]
    return pd.Series(vals, index=_future_index(train, h))


def fc_ma(train, h, k=4):
    return pd.Series(train.iloc[-k:].mean(), index=_future_index(train, h))


def _fourier(index: pd.DatetimeIndex, K: int = 3) -> pd.DataFrame:
    t = index.isocalendar().week.astype(int).values + (index.dayofyear.values % 7) / 7.0
    cols = {}
    for k in range(1, K + 1):
        cols[f"s{k}"] = np.sin(2 * np.pi * k * t / 52.18)
        cols[f"c{k}"] = np.cos(2 * np.pi * k * t / 52.18)
    return pd.DataFrame(cols, index=index)


def fc_arima(train, h, order=(1, 1, 1), K=3):
    """ARIMA errors + Fourier regressors (dynamic harmonic regression): captures annual seasonality
    without the instability of a 52-period seasonal ARIMA on ~3 years of weekly data."""
    fut = _future_index(train, h)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ARIMA(train, exog=_fourier(train.index, K), order=order).fit()
        fc = m.forecast(h, exog=_fourier(fut, K))
    fc.index = fut
    return fc


def fc_xgb(train, h):
    f = make_features(train)
    X, y = f.dropna(), train.loc[f.dropna().index]
    model = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                             colsample_bytree=0.9, random_state=config.RANDOM_STATE, n_jobs=2)
    model.fit(X, y)
    hist = train.copy()
    out = []
    for d in _future_index(train, h):
        row = _features_for_next(hist, d)
        # 't' must continue the training index, which make_features already guarantees
        p = float(model.predict(row)[0])
        out.append(p)
        hist.loc[d] = p
        hist = hist.sort_index()
    return pd.Series(out, index=_future_index(train, h))


def fc_snaive_growth(train, h):
    """Seasonal naive scaled by the trailing 52-week vs previous 52-week growth ratio."""
    g = train.iloc[-SP:].sum() / train.iloc[-2 * SP:-SP].sum()
    return fc_snaive(train, h) * g


def _yoy_features(s: pd.Series) -> pd.DataFrame:
    """Features for the YoY-ratio model; every column at week t uses only weeks < t."""
    r = s / s.shift(SP)
    f = pd.DataFrame(index=s.index)
    for l in (1, 2, 3, 4, 8, 13):
        f[f"yoy_lag{l}"] = r.shift(l)
    f["yoy_roll4"] = r.shift(1).rolling(4).mean()
    f["yoy_roll13"] = r.shift(1).rolling(13).mean()
    f["lag52_rel"] = s.shift(SP) / s.shift(1).rolling(SP).mean()     # last year's seasonal position
    f["week_of_year"] = s.index.isocalendar().week.astype(int).values
    f["month"] = s.index.month
    return f


def fc_xgb_yoy(train, h):
    """XGBoost predicts log(y_t / y_{t-52}); forecast = last year's value x predicted ratio.

    Seasonality comes from the lag-52 anchor and growth from the learned ratio, so the model needs
    no trend extrapolation. Recursive for multi-step horizons.
    """
    f = _yoy_features(train)
    target = np.log(train / train.shift(SP))
    X = f.dropna()
    model = xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.9,
                             colsample_bytree=0.9, random_state=config.RANDOM_STATE, n_jobs=2)
    model.fit(X, target.loc[X.index])
    hist = train.copy()
    out = []
    for d in _future_index(train, h):
        ext = pd.concat([hist, pd.Series([np.nan], index=[d])])
        ratio = float(np.exp(model.predict(_yoy_features(ext).iloc[[-1]])[0]))
        p = float(hist.iloc[-SP] * ratio)
        out.append(p)
        hist.loc[d] = p
        hist = hist.sort_index()
    return pd.Series(out, index=_future_index(train, h))


FORECASTERS = {"Naive": fc_naive, "Seasonal naive": fc_snaive, "Seasonal naive + growth": fc_snaive_growth,
               "Moving average (4w)": fc_ma, "ARIMA + Fourier": fc_arima, "XGBoost (levels)": fc_xgb,
               "XGBoost (YoY ratio)": fc_xgb_yoy}


# ------------------------------- evaluation -------------------------------
def split_points(n: int, folds: int = config.VALIDATION_FOLDS, h: int = H) -> dict:
    """Chronological split: final test = last h weeks; validation origins precede the test block."""
    test_start = n - h
    origins = [test_start - h * (folds - i) for i in range(folds)]      # train ends BEFORE each origin
    return {"test_start": test_start, "val_origins": origins}


def evaluate(y: pd.Series, h: int = H, folds: int = config.VALIDATION_FOLDS) -> dict:
    sp = split_points(len(y), folds, h)
    rows, val_preds = [], {}
    for name, fn in FORECASTERS.items():
        # --- validation folds (used for selection) ---
        for k, o in enumerate(sp["val_origins"]):
            train, actual = y.iloc[:o], y.iloc[o:o + h]
            pred = fn(train, h)
            rows.append({"model": name, "stage": "validation", "fold": k + 1, **metrics(actual, pred)})
            val_preds[(name, k)] = (actual, pred)
        # --- final held-out test ---
        train, actual = y.iloc[:sp["test_start"]], y.iloc[sp["test_start"]:]
        pred = fn(train, h)
        rows.append({"model": name, "stage": "test", "fold": 0, **metrics(actual, pred)})
        val_preds[(name, "test")] = (actual, pred)
    res = pd.DataFrame(rows)
    sel = (res[res.stage == "validation"].groupby("model")[["MAE", "RMSE", "WAPE", "MAPE", "Bias"]].mean()
           .sort_values("WAPE"))
    best = sel.index[0]
    return {"results": res, "selection": sel.reset_index(), "best_model": best, "split": sp, "preds": val_preds}


def interval_ratios(preds: dict, model: str, folds: int = config.VALIDATION_FOLDS) -> tuple[float, float]:
    """10th / 90th percentile of actual/forecast across validation folds -> empirical 80% band."""
    r = np.concatenate([(preds[(model, k)][0].values / preds[(model, k)][1].values) for k in range(folds)])
    return float(np.quantile(r, 0.10)), float(np.quantile(r, 0.90))


def future_forecast(y: pd.Series, ev: dict, h: int = H) -> pd.DataFrame:
    best = ev["best_model"]
    fc = FORECASTERS[best](y, h)
    lo, hi = interval_ratios(ev["preds"], best)
    return pd.DataFrame({"week_ending": fc.index, "model": best, "revenue_base": fc.values,
                         "revenue_low": fc.values * lo, "revenue_high": fc.values * hi})


def category_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Same protocol, per category (validation mean + test WAPE)."""
    out = []
    for cat, d in df.groupby("category"):
        ev = evaluate(weekly_series(d))
        val = ev["results"][ev["results"].stage == "validation"].groupby("model").WAPE.mean()
        tst = ev["results"][ev["results"].stage == "test"].set_index("model").WAPE
        for m in MODELS:
            out.append({"category": cat, "model": m, "validation_WAPE": val[m], "test_WAPE": tst[m],
                        "selected": m == ev["best_model"]})
    return pd.DataFrame(out)


def plausibility(y: pd.Series, fut: pd.DataFrame) -> dict:
    """Sanity guard: compare the next-13-week forecast with the same weeks one year earlier."""
    last_year = float(y.iloc[-SP:-SP + len(fut)].sum())
    base = float(fut.revenue_base.sum())
    trailing_growth = float(y.iloc[-SP:].sum() / y.iloc[-2 * SP:-SP].sum() - 1)
    return {"same_weeks_last_year": last_year, "forecast_base": base, "ratio": base / last_year,
            "trailing_52w_growth": trailing_growth}
