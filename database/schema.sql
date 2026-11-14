-- Star schema for the Sales Intelligence platform (SQLite; portable to MySQL/PostgreSQL).
PRAGMA foreign_keys = ON;
DROP VIEW IF EXISTS v_monthly_kpis;
DROP TABLE IF EXISTS fact_sales;
DROP TABLE IF EXISTS dim_customer;
DROP TABLE IF EXISTS dim_product;
DROP TABLE IF EXISTS dim_region;
DROP TABLE IF EXISTS dim_date;

CREATE TABLE dim_date (
  date_key INTEGER PRIMARY KEY, date TEXT NOT NULL, year INTEGER, quarter INTEGER,
  month INTEGER, month_start TEXT, weekday INTEGER);
CREATE TABLE dim_customer (
  customer_key INTEGER PRIMARY KEY, customer_id TEXT UNIQUE NOT NULL, customer_name TEXT, segment TEXT);
CREATE TABLE dim_product (
  product_key INTEGER PRIMARY KEY, product_id TEXT NOT NULL, product_name TEXT NOT NULL,
  category TEXT, sub_category TEXT, UNIQUE (product_id, product_name));
CREATE TABLE dim_region (
  region_key INTEGER PRIMARY KEY, market TEXT, region TEXT, country TEXT, state TEXT, city TEXT,
  UNIQUE (market, region, country, state, city));
CREATE TABLE fact_sales (
  row_id INTEGER PRIMARY KEY, order_key TEXT NOT NULL, order_id TEXT NOT NULL,
  order_date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
  ship_date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
  customer_key INTEGER NOT NULL REFERENCES dim_customer(customer_key),
  product_key INTEGER NOT NULL REFERENCES dim_product(product_key),
  region_key INTEGER NOT NULL REFERENCES dim_region(region_key),
  ship_mode TEXT, order_priority TEXT, quantity INTEGER, sales_amount REAL, discount REAL,
  profit REAL, shipping_cost REAL);
CREATE INDEX ix_fact_order_date ON fact_sales(order_date_key);
CREATE INDEX ix_fact_customer ON fact_sales(customer_key);
CREATE INDEX ix_fact_product ON fact_sales(product_key);

CREATE VIEW v_monthly_kpis AS
SELECT d.month_start AS month,
       SUM(f.sales_amount) AS revenue, SUM(f.profit) AS profit,
       COUNT(DISTINCT f.order_key) AS orders, SUM(f.quantity) AS units,
       SUM(f.sales_amount) / COUNT(DISTINCT f.order_key) AS avg_order_value,
       SUM(f.profit) / SUM(f.sales_amount) AS profit_margin
FROM fact_sales f JOIN dim_date d ON d.date_key = f.order_date_key
GROUP BY d.month_start;
