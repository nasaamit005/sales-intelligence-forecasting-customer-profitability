import sqlite3

import pytest

from salesintel import config, database


@pytest.fixture(scope="module")
def con(pipe):
    c = sqlite3.connect(config.DB_FILE)
    yield c
    c.close()


def test_sql_reconciles_with_pandas(con, df):
    assert database.reconcile(con, df).match.all()


def test_foreign_keys_intact(con):
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []


def test_fact_rows_equal_clean_rows(con, df):
    assert con.execute("SELECT COUNT(*) FROM fact_sales").fetchone()[0] == len(df)


def test_category_profit_matches(con, df):
    sql = database.run_query(con, "category_profitability").groupby("category").profit.sum()
    py = df.groupby("category").profit.sum()
    assert (sql - py).abs().max() < 1e-6


def test_monthly_view_matches(con, df):
    v = con.execute("SELECT SUM(revenue), COUNT(*) FROM v_monthly_kpis").fetchone()
    assert abs(v[0] - df.sales.sum()) < 1e-3 and v[1] == df.month.nunique()


def test_forecast_fact_table_present(con):
    assert con.execute("SELECT COUNT(*) FROM fact_forecast").fetchone()[0] > 0
