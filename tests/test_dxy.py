import pandas as pd
from debco.data.dxy import build_dxy_from_components


def test_build_dxy_from_components_constant_prices():
    dates = pd.date_range("2024-01-01", periods=3, freq="15min")
    comps = {
        "EURUSD": pd.DataFrame({"date": dates, "close": [1.1, 1.1, 1.1]}),
        "USDJPY": pd.DataFrame({"date": dates, "close": [150, 150, 150]}),
        "GBPUSD": pd.DataFrame({"date": dates, "close": [1.25, 1.25, 1.25]}),
        "USDCAD": pd.DataFrame({"date": dates, "close": [1.35, 1.35, 1.35]}),
        "USDSEK": pd.DataFrame({"date": dates, "close": [10.5, 10.5, 10.5]}),
        "USDCHF": pd.DataFrame({"date": dates, "close": [0.9, 0.9, 0.9]}),
    }
    out = build_dxy_from_components(comps)
    assert len(out) == 3
    assert "dxy_close" in out.columns
    assert "dxy_inverse_close" in out.columns
    assert "index_close" in out.columns
