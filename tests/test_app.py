import pytest

PAGES = ["/", "/sales", "/products", "/regions", "/customers", "/forecasting", "/discounts", "/planning", "/reports"]


@pytest.mark.parametrize("url", PAGES)
def test_pages_render(client, url):
    assert client.get(url).status_code == 200


def test_health_matches_data(client, df):
    j = client.get("/health").get_json()
    assert j["rows"] == len(df) and abs(j["revenue"] - df.sales.sum()) < 1e-3


def test_filters_change_and_reconcile_kpis(client, df):
    html = client.get("/?year=2014").get_data(as_text=True)
    assert f"${df[df.year == 2014].sales.sum():,.0f}" in html
    assert f"${df.sales.sum():,.0f}" not in html


def test_impossible_filter_has_empty_state(client, df):
    from web.views import apply_filters
    out, _ = apply_filters(df, {"year": "2011", "market": "Nowhere"})
    assert out.empty
    assert "No data matches" in client.get("/?market=Nowhere").get_data(as_text=True)


def test_csv_export_respects_filters(client, df):
    all_rows = client.get("/export/monthly.csv").get_data(as_text=True).count("\n")
    y2014 = client.get("/export/monthly.csv?year=2014").get_data(as_text=True).count("\n")
    assert y2014 < all_rows


def test_downloads_and_404(client):
    assert client.get("/download/excel").status_code == 200
    assert client.get("/download/unknown").status_code == 404
    assert client.get("/missing-page").status_code == 404
