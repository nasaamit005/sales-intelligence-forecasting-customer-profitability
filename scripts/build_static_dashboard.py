"""Build a self-contained interactive HTML dashboard from the pipeline outputs.
Usage: python scripts/build_static_dashboard.py [output.html]"""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from salesintel import analytics as A, commercial as C, config, data  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports" / "html" / "dashboard.html"
df = data.load_clean(); R = config.RESULTS_DIR; M = json.loads((R / "metrics.json").read_text()); fm = M["forecast"]
cat_of = df.drop_duplicates("sub_category").set_index("sub_category").category.to_dict()
subs = sorted(cat_of); cats = sorted(set(cat_of.values())); mkts = sorted(df.market.unique()); bands = config.DISCOUNT_LABELS
g = df.assign(mo=df.order_date.dt.month).groupby(["year", "mo", "market", "sub_category", "discount_band"]).agg(
    rev=("sales", "sum"), prof=("profit", "sum"), units=("quantity", "sum"), lines=("sales", "size")).reset_index()
F = [[int(r.year), int(r.mo), mkts.index(r.market), subs.index(r.sub_category), bands.index(r.discount_band),
      round(r.rev), round(r.prof), int(r.units), int(r.lines)] for r in g.itertuples()]
rf = A.rfm(df); seg = A.segment_profile(rf); coh = A.cohort_retention(df); cv = A.customer_value(df).head(10)
pareto = A.pareto(df); k = A.kpis(df)
wk = pd.read_csv(R / "weekly_revenue.csv").tail(110)
pred = pd.read_csv(R / "forecast_predictions.csv"); best = fm["best_model"]
tp = pred[(pred.model_name == best) & (pred.stage == "test")]
fut = pd.read_csv(R / "forecast_next13.csv"); plan = pd.read_csv(R / "demand_plan.csv"); hz = C.horizon_summary(plan)
ce = pd.read_csv(R / "category_forecast_evaluation.csv"); weak = pd.read_csv(R / "weak_profit_products.csv").head(8)
db = C.discount_bands(df); opp = C.opportunity_matrix(df); sm = seg.set_index("segment")
b31 = M["discount"]; pl = fm["plausibility"]; ch = M["churn"]
invest = ", ".join(opp[opp.quadrant.str.startswith("Invest")].sort_values("revenue", ascending=False).sub_category.head(4))
tables = df.groupby("sub_category").profit.sum()["Tables"]
finds = [
 f"Revenue grew every year: ${M['revenue_by_year']['2011']/1e6:.2f}M (2011) to ${M['revenue_by_year']['2014']/1e6:.2f}M (2014), +{M['revenue_by_year']['2014']/M['revenue_by_year']['2013']-1:.1%} in 2014. Reported margin is {M['margin']:.1%}.",
 f"Customer concentration: the top 20% of customers generate {pareto['top20pct_revenue_share']:.0%} of revenue; the {int(sm.loc['Champions','customers'])} ‘Champions’ alone hold {sm.loc['Champions','revenue_share']:.0%}.",
 f"Discounts are where margin disappears: full-price lines earn {b31['margin_0pct']:.1%}, while lines discounted 31% or more are {b31['revenue_share_31_plus']:.1%} of revenue at a {b31['margin_31_plus']:.1%} reported margin, and {b31['loss_line_share_31_plus']:.0%} of them lose money (association, not proven cause).",
 f"{M['weak_profit_products']} products with more than $5,000 revenue earn under a 2% reported margin. The Tables sub-category loses ${-tables/1e3:.0f}k on $757k revenue.",
 f"EMEA is the weakest large market at a 5.4% reported margin; Canada is tiny ($67k revenue) but the highest-margin market.",
 f"Churn model (time-split, 180-day lapse): AUC {ch['auc_logistic']:.2f} vs {ch['auc_recency_baseline']:.2f} for a recency-only rule. Purchase frequency is the strongest signal.",
 f"Forecast: best validation model is {best} (validation WAPE {fm['selection'][best]['WAPE']:.1%}). Next 13 weeks base case ${fm['next13_base_revenue']/1e3:,.0f}k, which is {pl['ratio']:.2f}x the same weeks last year (trailing 52-week growth {pl['trailing_52w_growth']:.0%}).",
]
recs = [
 "Introduce approval rules for discounts above 30%: that band is about 12% of revenue and runs at a negative reported margin. Test the rule on a controlled sample before rolling it out, because this data cannot prove cause.",
 f"Review the {M['weak_profit_products']} low-margin products and the Tables sub-category for price, cost or discount changes.",
 f"Protect the {int(sm.loc['Champions','customers'])} Champions and run a retention offer for the {int(sm.loc['At Risk','customers'])} ‘At Risk’ customers ({sm.loc['At Risk','revenue_share']:.0%} of revenue), prioritised by the churn scores.",
 f"Plan Q1 2015 supply around the base case (${fm['next13_base_revenue']/1e3:,.0f}k) with the low/high range (${fm['next13_low']/1e3:,.0f}k to ${fm['next13_high']/1e3:,.0f}k) as scenarios.",
 f"Prioritise growth sub-categories in the ‘Invest’ quadrant (growing and above-median margin): {invest}.",
]
TPM = {m: [[str(a)[:10], round(b)] for a, b in zip(x.forecast_date, x.predicted)] for m, x in pred[pred.stage == 'test'].groupby('model_name')}
D = dict(F=F, M=mkts, S=subs, C=cats, SC=[cats.index(cat_of[s]) for s in subs], B=bands,
 kp=dict(orders=int(k["orders"]), customers=int(k["customers"]), aov=round(k["aov"]), top20=round(pareto["top20pct_revenue_share"], 3), rows=len(df)),
 loss=[round(float(x), 3) for x in db.loss_line_share],
 seg=[[r.segment, int(r.customers), round(r.avg_recency), round(r.avg_frequency, 1), round(r.revenue_share, 4), round(r.margin, 4)] for r in seg.itertuples()],
 coh=dict(q=coh.cohort.tolist(), n=coh.cohort_size.tolist(), r=[[None if pd.isna(v) else round(float(v), 2) for v in row] for row in coh.drop(columns=["cohort", "cohort_size"]).values]),
 top=[[r.customer_name, r.segment, int(r.orders), round(r.revenue), round(r.profit)] for r in cv.itertuples()],
 wk=[[str(a)[:10], round(b)] for a, b in zip(wk.week_ending, wk.revenue)], tp=[[str(a)[:10], round(b)] for a, b in zip(tp.forecast_date, tp.predicted)],
 fut=[[str(r.week_ending)[:10], round(r.revenue_base), round(r.revenue_low), round(r.revenue_high)] for r in fut.itertuples()], best=best, tpm=TPM,
 sel=[[m, v["WAPE"], v["MAE"], fm["test"][m]["WAPE"], fm["test"][m]["MAE"]] for m, v in sorted(fm["selection"].items(), key=lambda kv: kv[1]["WAPE"])],
 cat=[[r.category, r.model, round(r.validation_WAPE, 3), round(r.test_WAPE, 3), bool(r.selected)] for r in ce.itertuples()],
 hz=[[r.horizon, round(r.revenue_low), round(r.revenue_base), round(r.revenue_high), round(r.units_low), round(r.units_base), round(r.units_high)] for r in hz.itertuples()],
 weak=[[r.product_name, r.sub_category, round(r.revenue), round(r.margin, 3), round(r.avg_discount, 3)] for r in weak.itertuples()],
 churn=dict(auc=round(ch["auc_logistic"], 3), base=round(ch["auc_recency_baseline"], 3), tr=round(ch["train_churn_rate"], 3), te=round(ch["test_churn_rate"], 3)),
 pl=dict(ratio=round(pl["ratio"], 2), g=round(pl["trailing_52w_growth"], 3)), finds=finds, recs=recs)
html = (ROOT / "scripts" / "dashboard_template.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(D, separators=(",", ":")).replace("</", "<\\/"))
out.parent.mkdir(parents=True, exist_ok=True); out.write_text(html, encoding="utf-8")
print(out, f"{len(html)/1024:.0f} KB", len(F), "fact rows")
