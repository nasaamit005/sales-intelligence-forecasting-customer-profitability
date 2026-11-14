"""Ingestion, cleaning and validation. Raw data is read-only; outputs go to data/processed."""
from __future__ import annotations

import pandas as pd

from . import config

RENAME = {
    "Row ID": "row_id", "Order ID": "order_id", "Order Date": "order_date", "Ship Date": "ship_date",
    "Ship Mode": "ship_mode", "Customer ID": "customer_id", "Customer Name": "customer_name",
    "Segment": "segment", "City": "city", "State": "state", "Country": "country",
    "Postal Code": "postal_code", "Market": "market", "Region": "region", "Product ID": "product_id",
    "Category": "category", "Sub-Category": "sub_category", "Product Name": "product_name",
    "Sales": "sales", "Quantity": "quantity", "Discount": "discount", "Profit": "profit",
    "Shipping Cost": "shipping_cost", "Order Priority": "order_priority",
}


def load_raw(path=config.RAW_FILE) -> pd.DataFrame:
    """Read the raw CSV exactly as supplied (never written back)."""
    return pd.read_csv(path, encoding="latin1")


def parse_mixed_dates(s: pd.Series) -> pd.Series:
    """The source mixes 'dd-mm-yyyy' and 'd/m/yyyy' strings, both DAY-first.

    Verified empirically: with day-first parsing every ship lag is 0-7 days; month-first
    parsing yields thousands of negative lags.
    """
    s = s.astype(str).str.strip()
    dash = s.str.contains("-", regex=False)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out[dash] = pd.to_datetime(s[dash], format="%d-%m-%Y")
    out[~dash] = pd.to_datetime(s[~dash], format="%d/%m/%Y")
    return out


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=RENAME).copy()
    df["order_date"] = parse_mixed_dates(df["order_date"])
    df["ship_date"] = parse_mixed_dates(df["ship_date"])
    df = df.drop_duplicates()
    # Order IDs are re-used across dates/customers in the source (659 IDs), so an order is
    # defined as the combination of order_id + order_date + customer_id.
    df["order_key"] = (df["order_id"] + "|" + df["order_date"].dt.strftime("%Y-%m-%d")
                       + "|" + df["customer_id"])
    # Product IDs map to several names in the source; the product key combines both.
    df["product_key"] = df["product_id"] + "|" + df["product_name"]
    df["ship_days"] = (df["ship_date"] - df["order_date"]).dt.days
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    df["quarter"] = df["order_date"].dt.quarter
    df["discount_band"] = pd.cut(df["discount"], bins=config.DISCOUNT_BINS,
                                 labels=config.DISCOUNT_LABELS).astype(str)
    df["list_value"] = df["sales"] / (1 - df["discount"].clip(upper=0.99))  # est. pre-discount value
    df["profit_margin"] = df["profit"] / df["sales"]
    df["postal_code"] = df["postal_code"].astype("string")
    return df.sort_values(["order_date", "row_id"]).reset_index(drop=True)


def validate(df: pd.DataFrame, raw_rows: int | None = None) -> pd.DataFrame:
    """Return a data-quality report. Each check has a pass flag and a detail string."""
    checks = []

    def add(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("Row ID unique", df["row_id"].is_unique, f"{len(df):,} rows")
    if raw_rows is not None:
        add("No rows lost in cleaning", len(df) == raw_rows, f"raw={raw_rows:,}, clean={len(df):,}")
    add("Dates parsed", df[["order_date", "ship_date"]].notna().all().all(),
        f"{df.order_date.min().date()} to {df.order_date.max().date()}")
    add("Ship date not before order date", (df["ship_days"] >= 0).all(),
        f"ship lag {int(df.ship_days.min())}-{int(df.ship_days.max())} days")
    add("Sales > 0", (df["sales"] > 0).all(), f"min={df.sales.min():.2f}")
    add("Quantity > 0", (df["quantity"] > 0).all(), f"min={df.quantity.min()}")
    add("Discount within [0, 1]", df["discount"].between(0, 1).all(),
        f"max={df.discount.max():.2f}")
    add("Shipping cost >= 0", (df["shipping_cost"] >= 0).all(), f"min={df.shipping_cost.min():.2f}")
    add("No missing keys", df[["order_id", "customer_id", "product_id"]].notna().all().all(), "")
    multi_date = int((df.groupby("order_id")["order_date"].nunique() > 1).sum())
    add("Order ID unique per order (documented exception)", multi_date == 0,
        f"{multi_date} order IDs reused on different dates/customers -> order_key used")
    names_per_id = int((df.groupby("customer_name")["customer_id"].nunique() > 1).sum())
    add("Customer name unique per customer ID (documented exception)", names_per_id == 0,
        f"{names_per_id} names shared by several IDs -> customer_id used as key")
    prod_multi = int((df.groupby("product_id")["product_name"].nunique() > 1).sum())
    add("Product ID unique per product name (documented exception)", prod_multi == 0,
        f"{prod_multi} product IDs carry several names -> product_key used")
    add("Loss-making line share (informational)", True,
        f"{(df.profit < 0).mean():.1%} of lines have negative reported profit")
    add("Postal code missing (informational)", True,
        f"{df.postal_code.isna().mean():.1%} missing; not used in analysis")
    add("Returns / cancellations identifiable", False,
        "no return or cancellation field exists; all lines treated as completed sales")
    return pd.DataFrame(checks)


def build_clean_dataset(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_raw()
    df = clean(raw)
    report = validate(df, raw_rows=len(raw))
    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.CLEAN_FILE, index=False)
        report.to_csv(config.PROCESSED_DIR / "data_quality_report.csv", index=False)
    return df, report


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(config.CLEAN_FILE, parse_dates=["order_date", "ship_date", "month"],
                     dtype={"postal_code": "string"})
    return df
