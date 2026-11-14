import openpyxl

from salesintel import commercial as C
from salesintel import reporting


def test_discount_bands_cover_all_revenue(df):
    assert abs(C.discount_bands(df).revenue.sum() - df.sales.sum()) < 1e-3
    assert C.discount_bands(df).lines.sum() == len(df)


def test_demand_scenarios_ordered(pipe):
    p = pipe["plan"]
    assert (p.units_low <= p.units_base).all() and (p.units_base <= p.units_high).all()
    s = C.horizon_summary(p)
    assert s.units_base.is_monotonic_increasing and len(s) == 3


def test_opportunity_quadrants_cover_every_subcategory(df):
    o = C.opportunity_matrix(df)
    assert len(o) == df.sub_category.nunique() and o.quadrant.notna().all()


def test_excel_workbook_matches_pipeline(pipe, tmp_path):
    p = reporting.build_excel(pipe["tables"], tmp_path / "r.xlsx")
    wb = openpyxl.load_workbook(p, data_only=True)
    assert len(wb.sheetnames) == 17
    ws = wb["Executive Dashboard"]
    assert abs(ws["B4"].value - pipe["metrics"]["revenue"]) < 1e-3
    assert abs(ws["B5"].value - pipe["metrics"]["profit"]) < 1e-3


def test_html_report_contains_definitions_and_limitations(pipe, tmp_path):
    p = reporting.build_html(pipe["tables"], tmp_path / "r.html")
    txt = p.read_text()
    assert "Limitations" in txt and f"{pipe['metrics']['orders']:,}" in txt
