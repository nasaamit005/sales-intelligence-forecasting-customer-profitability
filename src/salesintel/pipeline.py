"""End-to-end pipeline: data -> SQL -> analytics -> forecasting -> commercial -> reports."""
from __future__ import annotations

import json

import pandas as pd

from . import analytics as A
from . import commercial as C
from . import config, data, database, reporting
from . import forecasting as F


def _forecast_fact(ev: dict) -> pd.DataFrame:
    rows = []
    for (model, stage), (actual, pred) in ev["preds"].items():
        s = "test" if stage == "test" else f"validation_fold_{stage + 1}"
        rows.append(pd.DataFrame({"forecast_date": actual.index, "model_name": model, "stage": s,
                                  "actual": actual.values, "predicted": pred.values}))
    return pd.concat(rows, ignore_index=True)


def run(save: bool = True, build_reports: bool = True, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    log("1/6 ingest, clean, validate")
    df, dq = data.build_clean_dataset(save=save)
    log("2/6 relational database + reconciliation")
    con = database.build_database(df)
    recon = database.reconcile(con, df)
    assert recon.match.all(), "SQL and pandas totals disagree"

    log("3/6 analytics (sales, products, regions, customers, churn)")
    k = A.kpis(df)
    rfm = A.rfm(df)
    churn = A.churn_model(df)
    yearly = A.by_dimension(df, "year").sort_values("year")
    t = {"kpis": k, "monthly": A.monthly(df), "yearly": yearly, "products": A.top_bottom_products(df),
         "subcat": A.by_dimension(df, ["category", "sub_category"]), "market": A.by_dimension(df, ["market", "year"]),
         "segments": A.segment_profile(rfm), "rfm": rfm, "cohort": A.cohort_retention(df),
         "cust_value": A.customer_value(df), "dq": dq,
         "scope_note": f"{df.order_date.min():%d %b %Y} to {df.order_date.max():%d %b %Y}; {len(df):,} order lines."}

    log("4/6 forecasting (5 models, rolling-origin validation, held-out test)")
    y = F.weekly_series(df)
    ev = F.evaluate(y)
    fut = F.future_forecast(y, ev)
    cat_eval = F.category_evaluation(df)
    fact_fc = _forecast_fact(ev)

    log("5/6 commercial analytics")
    plaus = F.plausibility(y, fut)
    plan = C.demand_planning(fut, df)
    t.update({"selection": ev["selection"], "forecast_results": ev["results"], "discount_bands": C.discount_bands(df),
              "horizon": C.horizon_summary(plan)})
    test_tbl = ev["results"][ev["results"].stage == "test"][["model", "MAE", "RMSE", "WAPE", "MAPE", "Bias"]].sort_values("WAPE")
    t["test_table"] = test_tbl

    pc = A.pareto(df)
    bands = t["discount_bands"].set_index("discount_band")
    big = bands.loc[["31-50%", ">50%"]]
    metrics = {
        "rows": len(df), "orders": k["orders"], "customers": k["customers"], "revenue": k["revenue"],
        "profit": k["profit"], "margin": k["margin"], "date_min": str(df.order_date.min().date()),
        "date_max": str(df.order_date.max().date()), "weeks": len(y), "top20_revenue_share": pc["top20pct_revenue_share"],
        "revenue_by_year": df.groupby("year").sales.sum().to_dict(),
        "forecast": {"best_model": ev["best_model"], "split": {k2: (v if not isinstance(v, list) else v) for k2, v in ev["split"].items()},
                     "selection": ev["selection"].set_index("model").round(4).to_dict("index"),
                     "test": test_tbl.set_index("model").round(4).to_dict("index"),
                     "next13_base_revenue": float(fut.revenue_base.sum()),
                     "next13_low": float(fut.revenue_low.sum()), "next13_high": float(fut.revenue_high.sum()),
                     "plausibility": plaus},
        "churn": {k2: v for k2, v in churn.items() if k2 not in ("scored", "coefficients")},
        "discount": {"margin_0pct": float(bands.loc["0%", "margin"]), "margin_31_plus": float(big.profit.sum() / big.revenue.sum()),
                     "revenue_share_31_plus": float(big.revenue.sum() / bands.revenue.sum()),
                     "loss_line_share_31_plus": float((big.loss_line_share * big.lines).sum() / big.lines.sum())},
        "weak_profit_products": int(len(C.weak_profit_products(df))),
        "sql_reconciliation": recon.to_dict("records"),
    }
    sm = t["segments"].set_index("segment")
    t["findings"] = [
        f"Revenue grew every year, from ${metrics['revenue_by_year'][2011]/1e6:.2f}M in 2011 to ${metrics['revenue_by_year'][2014]/1e6:.2f}M in 2014; reported margin is {k['margin']:.1%}.",
        f"The top 20% of customers generate {pc['top20pct_revenue_share']:.0%} of revenue; Champions alone hold {sm.loc['Champions','revenue_share']:.0%}.",
        f"Lines discounted by 31% or more are {metrics['discount']['revenue_share_31_plus']:.0%} of revenue but run at a {metrics['discount']['margin_31_plus']:.0%} reported margin versus {metrics['discount']['margin_0pct']:.0%} at full price (descriptive, not causal).",
        f"{metrics['weak_profit_products']} products with >$5,000 revenue earn under a 2% reported margin and merit pricing review.",
        f"Best forecast model: {ev['best_model']} (held-out test WAPE {metrics['forecast']['test'][ev['best_model']]['WAPE']:.1%}). Base-case revenue for the next 13 weeks: ${fut.revenue_base.sum():,.0f} ({plaus['ratio']:.2f}x the same weeks last year; trailing 52-week growth {plaus['trailing_52w_growth']:.0%}).",
        f"A logistic churn model reaches AUC {churn['auc_logistic']:.2f} on a time-split test (recency-only baseline {churn['auc_recency_baseline']:.2f}).",
    ]
    t["metrics"] = metrics

    if save:
        R = config.RESULTS_DIR
        ev["results"].to_csv(R / "forecast_evaluation.csv", index=False)
        ev["selection"].to_csv(R / "model_selection.csv", index=False)
        fut.to_csv(R / "forecast_next13.csv", index=False)
        plan.to_csv(R / "demand_plan.csv", index=False)
        cat_eval.to_csv(R / "category_forecast_evaluation.csv", index=False)
        fact_fc.to_csv(R / "forecast_predictions.csv", index=False)
        y.rename("revenue").rename_axis("week_ending").to_csv(R / "weekly_revenue.csv")
        churn["scored"].to_csv(R / "churn_scores_2014H2.csv", index=False)
        churn["coefficients"].to_csv(R / "churn_coefficients.csv", index=False)
        C.opportunity_matrix(df).to_csv(R / "opportunity_matrix.csv", index=False)
        C.weak_profit_products(df).to_csv(R / "weak_profit_products.csv", index=False)
        C.discount_by_subcategory(df).to_csv(R / "discount_by_subcategory.csv", index=False)
        fact_fc.to_sql("fact_forecast", con, if_exists="replace", index=False)
        (R / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
        if build_reports:
            log("6/6 Excel workbook + HTML report")
            reporting.build_excel(t)
            reporting.build_html(t)
    con.close()
    log("done")
    return {"df": df, "tables": t, "evaluation": ev, "forecast": fut, "plan": plan, "churn": churn,
            "category_eval": cat_eval, "metrics": metrics}
