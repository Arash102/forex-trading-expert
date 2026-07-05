from __future__ import annotations

import pandas as pd

from debco.ml.candidates import candidate_mask_for_job
from debco.ml.setup_inventory import list_setup_specs


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_block_id": [4, 3, 6, 5],
            "distance_from_asia_high_pips": [4.0, -5.0, 1.0, 2.0],
            "distance_from_asia_low_pips": [10.0, 5.0, 1.0, -300.0],
            "distance_from_confirmed_zigzag_low_pips": [10.0, 8.0, 30.0, 300.0],
            "distance_from_confirmed_zigzag_high_pips": [-10.0, -200.0, -20.0, -100.0],
            "current_session_return_so_far_pips": [5.0, -2.0, -1.0, -250.0],
            "london_expansion_vs_asia": [0.5, 0.2, 0.1, 0.3],
            "rsi": [55.0, 48.0, 66.0, 52.0],
            "gmma_distance": [12.0, -5.0, 20.0, -20.0],
            "gmma_distance_slope": [1.0, -2.0, 0.0, -3.0],
            "atr_regime": [2, 2, 3, 3],
            "h1_trend_direction": [1, 1, -1, 0],
            "body_ratio": [0.2, 0.15, 0.02, -0.1],
            "market_regime": [1, 0, 2, 1],
            "day_of_week": [2, 2, 1, 2],
            "spread_pips": [0.8, 0.8, 0.8, 50.0],
            "session_volatility_percentile_240": [0.3, 0.3, 0.4, 0.4],
            "dxy_inverse_zscore_20": [0.2, -0.2, -0.5, 0.1],
        }
    )


def test_setup_specific_mask_is_side_aware() -> None:
    df = _base_df()
    cfg = {
        "candidate_filter": {
            "enabled": True,
            "preset": "setup_specific_v1",
            "setup_id": "EUR_AH_ATR2_BUY",
            "base": {"max_spread_pips_by_symbol": {"EURUSD": 1.3}},
        }
    }
    long_mask = candidate_mask_for_job(df, symbol="EURUSD", side="long", config=cfg)
    short_mask = candidate_mask_for_job(df, symbol="EURUSD", side="short", config=cfg)
    assert bool(long_mask.iloc[0])
    assert not bool(short_mask.any())


def test_setup_inventory_specs_parse_config() -> None:
    cfg = {
        "setup_inventory": {
            "setups": [
                {
                    "enabled": True,
                    "setup_id": "EUR_AH_ATR2_BUY",
                    "symbol": "EURUSD",
                    "side": "long",
                    "profile": "fast_15_8_h16",
                    "family": "eurusd_buy",
                    "candidate_filter": {"preset": "setup_specific_v1"},
                }
            ]
        }
    }
    specs = list_setup_specs(cfg)
    assert len(specs) == 1
    assert specs[0].job.name == "EURUSD_fast_15_8_h16_long"
    assert specs[0].candidate_filter["setup_id"] == "EUR_AH_ATR2_BUY"


def test_redesign_setup_mask_routes_correctly() -> None:
    df = _base_df()
    cfg = {
        "candidate_filter": {
            "enabled": True,
            "preset": "setup_specific_v1",
            "setup_id": "EUR_SELL_LONDON_BREAKDOWN",
            "base": {"max_spread_pips_by_symbol": {"EURUSD": 1.3}},
        }
    }
    short_mask = candidate_mask_for_job(df, symbol="EURUSD", side="short", config=cfg)
    long_mask = candidate_mask_for_job(df, symbol="EURUSD", side="long", config=cfg)
    assert short_mask.dtype == bool
    assert not bool(long_mask.any())


def test_xau_redesign_setup_mask_routes_correctly() -> None:
    df = _base_df()
    cfg = {
        "candidate_filter": {
            "enabled": True,
            "preset": "setup_specific_v1",
            "setup_id": "XAU_SELL_H1DOWN_CONT",
            "base": {"max_spread_pips_by_symbol": {"XAUUSD": 100}},
        }
    }
    short_mask = candidate_mask_for_job(df, symbol="XAUUSD", side="short", config=cfg)
    wrong_side = candidate_mask_for_job(df, symbol="XAUUSD", side="long", config=cfg)
    assert short_mask.dtype == bool
    assert not bool(wrong_side.any())


def test_new_xau_redesign_setup_mask_routes_correctly() -> None:
    df = _base_df()
    cfg = {
        "candidate_filter": {
            "enabled": True,
            "preset": "setup_specific_v1",
            "setup_id": "XAU_SELL_DXY_PRESSURE",
            "base": {"max_spread_pips_by_symbol": {"XAUUSD": 100}},
        }
    }
    short_mask = candidate_mask_for_job(df, symbol="XAUUSD", side="short", config=cfg)
    wrong_side = candidate_mask_for_job(df, symbol="XAUUSD", side="long", config=cfg)
    assert short_mask.dtype == bool
    assert not bool(wrong_side.any())
