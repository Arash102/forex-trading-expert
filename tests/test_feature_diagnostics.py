from __future__ import annotations

from debco.live.feature_diagnostics import classify_feature_source, likely_source_for_bad_features


def test_classify_dxy_features() -> None:
    assert classify_feature_source("dxy_inverse_return_1") == "DXY"
    assert classify_feature_source("dxy_inverse_zscore_20") == "DXY"
    assert classify_feature_source("market_x") == "MARKET/DXY_RELATIVE_FEATURE"
    assert classify_feature_source("market_y") == "MARKET/DXY_RELATIVE_FEATURE"


def test_likely_source_detects_dxy_time_lag() -> None:
    msg = likely_source_for_bad_features(
        bad_features=["market_x", "dxy_inverse_return_1"],
        missing_features=[],
        dxy_last_time="2026-07-08T14:15:00Z",
        closed_bar_time="2026-07-08T14:30:00Z",
        has_dxy_exact_bar=False,
    )
    assert msg.startswith("DXY_TIME_LAG")
