-- name: total_kpis
SELECT SUM(sales_amount) AS revenue, SUM(profit) AS profit,
       COUNT(DISTINCT order_key) AS orders, SUM(quantity) AS units,
       COUNT(DISTINCT customer_key) AS customers
FROM fact_sales;

-- name: category_profitability
SELECT p.category, p.sub_category, SUM(f.sales_amount) AS revenue, SUM(f.profit) AS profit,
       SUM(f.profit) / SUM(f.sales_amount) AS margin
FROM fact_sales f JOIN dim_product p USING (product_key)
GROUP BY p.category, p.sub_category ORDER BY profit DESC;

-- name: market_year_revenue
SELECT r.market, d.year, SUM(f.sales_amount) AS revenue, SUM(f.profit) AS profit
FROM fact_sales f JOIN dim_region r USING (region_key)
JOIN dim_date d ON d.date_key = f.order_date_key
GROUP BY r.market, d.year ORDER BY r.market, d.year;

-- name: discount_band_profit
SELECT CASE WHEN discount = 0 THEN '0%' WHEN discount <= 0.10 THEN '1-10%'
            WHEN discount <= 0.20 THEN '11-20%' WHEN discount <= 0.30 THEN '21-30%'
            WHEN discount <= 0.50 THEN '31-50%' ELSE '>50%' END AS band,
       COUNT(*) AS lines, SUM(sales_amount) AS revenue, SUM(profit) AS profit,
       SUM(profit) / SUM(sales_amount) AS margin
FROM fact_sales GROUP BY band;

-- name: top_customers
SELECT c.customer_id, c.customer_name, c.segment, SUM(f.sales_amount) AS revenue,
       SUM(f.profit) AS profit, COUNT(DISTINCT f.order_key) AS orders
FROM fact_sales f JOIN dim_customer c USING (customer_key)
GROUP BY c.customer_id ORDER BY revenue DESC LIMIT 20;
