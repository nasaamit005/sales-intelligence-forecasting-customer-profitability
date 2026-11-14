import hashlib

import pandas as pd

from salesintel import config, data


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_mixed_date_formats_are_day_first():
    s = pd.Series(["13-01-2011", "1/1/2011", "6/1/2011", "31-12-2014"])
    out = data.parse_mixed_dates(s)
    assert out.tolist() == [pd.Timestamp("2011-01-13"), pd.Timestamp("2011-01-01"),
                            pd.Timestamp("2011-01-06"), pd.Timestamp("2014-12-31")]


def test_raw_file_never_modified():
    before = sha(config.RAW_FILE)
    data.clean(data.load_raw())
    assert sha(config.RAW_FILE) == before


def test_cleaning_keeps_all_rows_and_valid_ranges(df):
    raw = data.load_raw()
    assert len(df) == len(raw)
    assert df.ship_days.between(0, 7).all()
    assert df.discount.between(0, 1).all() and (df.sales > 0).all()


def test_critical_validation_checks_pass(df):
    rep = data.validate(df, raw_rows=len(data.load_raw()))
    must = ["Row ID unique", "No rows lost in cleaning", "Dates parsed", "Ship date not before order date",
            "Sales > 0", "Quantity > 0", "Discount within [0, 1]", "No missing keys"]
    got = rep.set_index("check").passed
    assert got[must].all()


def test_documented_exceptions_are_detected(df):
    rep = data.validate(df).set_index("check")
    assert not rep.loc["Order ID unique per order (documented exception)", "passed"]
    assert df.order_key.nunique() > df.order_id.nunique()      # order_key resolves re-used order IDs


def test_cleaning_is_rerunnable():
    a, _ = data.build_clean_dataset(save=False)
    b, _ = data.build_clean_dataset(save=False)
    pd.testing.assert_frame_equal(a, b)
