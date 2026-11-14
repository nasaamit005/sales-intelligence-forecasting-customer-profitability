"""Relational layer: loads the cleaned data into a star schema and runs the SQL library."""
from __future__ import annotations

import re
import sqlite3

import pandas as pd

from . import config


def _date_key(s: pd.Series) -> pd.Series:
    return s.dt.strftime("%Y%m%d").astype(int)


def build_database(df: pd.DataFrame, db_path=config.DB_FILE) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(config.SCHEMA_FILE.read_text())

    dates = pd.concat([df["order_date"], df["ship_date"]]).drop_duplicates().sort_values()
    dim_date = pd.DataFrame({
        "date_key": _date_key(dates), "date": dates.dt.strftime("%Y-%m-%d"), "year": dates.dt.year,
        "quarter": dates.dt.quarter, "month": dates.dt.month,
        "month_start": dates.dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m-%d"),
        "weekday": dates.dt.weekday})
    dim_customer = (df[["customer_id", "customer_name", "segment"]].drop_duplicates("customer_id")
                    .reset_index(drop=True).rename_axis("customer_key").reset_index())
    dim_customer["customer_key"] += 1
    dim_product = (df[["product_id", "product_name", "category", "sub_category"]]
                   .drop_duplicates(["product_id", "product_name"]).reset_index(drop=True)
                   .rename_axis("product_key").reset_index())
    dim_product["product_key"] += 1
    rcols = ["market", "region", "country", "state", "city"]
    dim_region = (df[rcols].drop_duplicates().reset_index(drop=True)
                  .rename_axis("region_key").reset_index())
    dim_region["region_key"] += 1

    f = df.merge(dim_customer[["customer_key", "customer_id"]], on="customer_id")
    f = f.merge(dim_product[["product_key", "product_id", "product_name"]], on=["product_id", "product_name"],
                suffixes=("_dropme", ""))
    f = f.merge(dim_region, on=rcols)
    fact = pd.DataFrame({
        "row_id": f["row_id"], "order_key": f["order_key"], "order_id": f["order_id"],
        "order_date_key": _date_key(f["order_date"]), "ship_date_key": _date_key(f["ship_date"]),
        "customer_key": f["customer_key"], "product_key": f["product_key"], "region_key": f["region_key"],
        "ship_mode": f["ship_mode"], "order_priority": f["order_priority"], "quantity": f["quantity"],
        "sales_amount": f["sales"], "discount": f["discount"], "profit": f["profit"],
        "shipping_cost": f["shipping_cost"]})
    for name, t in [("dim_date", dim_date), ("dim_customer", dim_customer), ("dim_product", dim_product),
                    ("dim_region", dim_region), ("fact_sales", fact)]:
        t.to_sql(name, con, if_exists="append", index=False)
    con.commit()
    return con


def load_queries() -> dict[str, str]:
    text = config.QUERIES_FILE.read_text()
    parts = re.split(r"-- name:\s*(\w+)\s*\n", text)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run_query(con: sqlite3.Connection, name: str) -> pd.DataFrame:
    return pd.read_sql_query(load_queries()[name], con)


def reconcile(con: sqlite3.Connection, df: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """Compare SQL aggregates against the pandas source (acceptance criterion)."""
    sql = run_query(con, "total_kpis").iloc[0]
    py = {"revenue": df.sales.sum(), "profit": df.profit.sum(), "orders": df.order_key.nunique(),
          "units": df.quantity.sum(), "customers": df.customer_id.nunique()}
    rows = [{"metric": k, "sql": float(sql[k]), "pandas": float(v),
             "match": abs(float(sql[k]) - float(v)) <= tol * max(1.0, abs(float(v)))} for k, v in py.items()]
    return pd.DataFrame(rows)
