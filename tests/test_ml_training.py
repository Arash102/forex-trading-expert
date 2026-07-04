from __future__ import annotations

import numpy as np
import pandas as pd

from debco.ml.calibration import expected_calibration_error, fit_probability_calibrator
from debco.ml.candidates import candidate_mask_for_job
from debco.ml.thresholding import threshold_sweep
from debco.ml.xgb_optuna import binary_classification_metrics
from debco.validation.cpcv import make_cpcv_splits
from debco.validation.walk_forward import make_walk_forward_splits


def test_binary_metrics_include_mcc() -> None:
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = binary_classification_metrics(y, proba, threshold=0.5)
    assert "mcc" in metrics
    assert metrics["mcc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_walk_forward_splits_are_causal_with_purge() -> None:
    folds = make_walk_forward_splits(
        100,
        train_window_bars=40,
        test_window_bars=10,
        step_bars=10,
        expanding=False,
        min_train_bars=30,
        purge_bars=5,
    )
    assert len(folds) > 0
    first = folds[0]
    assert first.train_idx.max() < first.test_idx.min()
    assert first.test_idx.min() - first.train_idx.max() == 6
    assert len(first.train_idx) == 35
    assert len(first.test_idx) == 10


def test_cpcv_splits_do_not_overlap_train_test_rows() -> None:
    dates = pd.date_range("2025-01-01", periods=60, freq="15min")
    meta = pd.DataFrame({"date": dates, "entry_date": dates, "exit_date": dates + pd.Timedelta(minutes=45)})
    folds = make_cpcv_splits(meta, n_groups=5, n_test_groups=2, embargo_bars=1, purge=True)
    assert len(folds) > 0
    for fold in folds:
        assert set(fold.train_idx).isdisjoint(set(fold.test_idx))
        assert len(fold.test_groups) == 2


def test_threshold_sweep_returns_signal_rates() -> None:
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.2, 0.6, 0.7, 0.9])
    out = threshold_sweep(y, proba, [0.5, 0.8], fold="x", probability_column="p")
    assert set(["threshold", "precision", "mcc", "signal_rate", "probability_column"]).issubset(out.columns)
    assert len(out) == 2
    assert out.loc[out["threshold"].eq(0.8), "signal_count"].iloc[0] == 1


def test_sigmoid_calibrator_and_ece_are_bounded() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.2, 0.35, 0.6, 0.8, 0.95])
    cal = fit_probability_calibrator(p, y, method="sigmoid")
    out = cal.predict(p)
    assert out.shape == p.shape
    assert np.all(out > 0)
    assert np.all(out < 1)
    ece = expected_calibration_error(y, out, bins=3)
    assert 0 <= ece <= 1


def test_candidate_filter_can_reduce_rows() -> None:
    x = pd.DataFrame({
        "session_block_id": [1, 3, 4, 6],
        "spread_pips": [1.0, 1.0, 2.0, 1.0],
        "gmma_distance": [0, 10, -30, 5],
        "atr_regime": [2, 2, 2, 5],
    })
    cfg = {
        "candidate_filter": {
            "enabled": True,
            "preset": "session_tradeable_v1",
            "session_block_ids": [3, 4, 6],
            "max_spread_pips_by_symbol": {"EURUSD": 1.5},
        }
    }
    mask = candidate_mask_for_job(x, symbol="EURUSD", side="long", config=cfg)
    assert mask.tolist() == [False, True, False, True]


def test_candidate_experiments_expand_to_named_configs() -> None:
    from debco.ml.candidates import list_candidate_experiments

    cfg = {
        "output": {"experiment_name": "base"},
        "candidate_experiments": {
            "enabled": True,
            "base_experiment_name": "candidate_base",
            "sets": [
                {"enabled": True, "name": "session vol", "candidate_filter": {"preset": "session_volatility_v1"}},
                {"enabled": False, "name": "skip", "candidate_filter": {"preset": "session_breakout_v1"}},
            ],
        },
    }
    exps = list_candidate_experiments(cfg)
    assert len(exps) == 1
    assert exps[0].name == "session_vol"
    assert exps[0].config["candidate_filter"]["enabled"] is True
    assert exps[0].config["output"]["experiment_name"] == "candidate_base_session_vol"


def test_session_breakout_candidate_filter_is_side_aware() -> None:
    from debco.ml.candidates import candidate_mask_for_job

    x = pd.DataFrame({
        "session_block_id": [3, 3, 3, 1],
        "spread_pips": [0.8, 0.8, 0.8, 0.8],
        "atr_percentile_240": [0.5, 0.5, 0.5, 0.5],
        "bar_volatility_vs_atr": [1.0, 1.0, 1.0, 1.0],
        "distance_from_asia_high_pips": [5.0, -1.0, 1.0, 10.0],
        "distance_from_asia_low_pips": [20.0, -5.0, -1.0, -10.0],
        "london_expansion_vs_asia": [1.0, 1.0, 0.2, 1.0],
        "current_session_return_so_far_pips": [3.0, -3.0, 0.0, 5.0],
    })
    cfg = {
        "candidate_filter": {
            "enabled": True,
            "preset": "session_breakout_v1",
            "base": {
                "session_block_ids": [3],
                "max_spread_pips_by_symbol": {"EURUSD": 1.5},
                "min_atr_percentile": 0.1,
                "max_bar_volatility_vs_atr": 3.0,
            },
            "session_breakout_v1": {
                "breakout_pips_by_symbol": {"EURUSD": 2.0},
                "min_london_expansion_vs_asia": 0.75,
                "min_current_session_return_pips_by_symbol": {"EURUSD": 1.0},
            },
        }
    }
    long_mask = candidate_mask_for_job(x, symbol="EURUSD", side="long", config=cfg)
    short_mask = candidate_mask_for_job(x, symbol="EURUSD", side="short", config=cfg)
    assert long_mask.tolist() == [True, False, False, False]
    assert short_mask.tolist() == [False, True, False, False]



def test_candidate_aware_folds_keep_base_timeline() -> None:
    import pandas as pd
    from debco.ml.xgb_optuna import build_candidate_aware_validation_folds

    cfg = {
        "validation": {
            "method": "walk_forward",
            "walk_forward": {
                "train_window_bars": 100,
                "test_window_bars": 20,
                "step_bars": 20,
                "min_train_bars": 80,
                "purge_bars": 0,
            },
        },
        "candidate_validation": {
            "mode": "base_timeline",
            "min_train_candidates": 10,
            "min_test_candidates": 2,
            "min_train_positives": 1,
            "min_test_positives": 1,
        },
    }
    metadata = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=180, freq="15min")})
    mask = pd.Series([i % 3 == 0 for i in range(180)])
    y = pd.Series([1 if i % 6 == 0 else 0 for i in range(180)])
    folds, stats = build_candidate_aware_validation_folds(cfg, metadata, mask, y)
    assert stats["base_fold_count"] > 1
    assert len(folds) > 1
    assert all(mask.iloc[f.test_idx].all() for f in folds)
    assert all(mask.iloc[f.train_idx].all() for f in folds)



def test_rule_inspired_core_routes_symbol_and_side() -> None:
    from debco.ml.candidates import candidate_mask_for_job

    x = pd.DataFrame({
        "session_block_id": [4, 3, 5, 5],
        "spread_pips": [1.0, 1.0, 50.0, 50.0],
        "market_regime": [1, 0, 1, 1],
        "distance_from_asia_high_pips": [4.0, 0.0, 0.0, 0.0],
        "distance_from_asia_low_pips": [20.0, 4.0, -10.0, -10.0],
        "london_expansion_vs_asia": [1.0, 0.0, 0.0, 0.0],
        "gmma_distance": [25.0, 0.0, -60.0, -10.0],
        "gmma_distance_slope": [1.0, 0.0, 1.0, -1.0],
        "body_ratio": [0.1, 0.2, 0.2, -0.1],
        "day_return_from_open_pips": [5.0, 0.0, 0.0, 0.0],
        "atr_regime": [2, 2, 2, 2],
        "rsi": [50.0, 50.0, 35.0, 50.0],
        "h1_trend_direction": [1, 1, 1, -1],
        "session_volatility_percentile_240": [0.5, 0.5, 0.5, 0.2],
        "day_of_week": [2, 2, 2, 2],
        "distance_from_confirmed_zigzag_low_pips": [30.0, 5.0, 0.0, 0.0],
    })
    cfg = {"candidate_filter": {"enabled": True, "preset": "rule_inspired_core_v1", "base": {"max_spread_pips_by_symbol": {"EURUSD": 1.3, "XAUUSD": 80.0}}}}
    eur_long = candidate_mask_for_job(x, symbol="EURUSD", side="long", config=cfg)
    xau_long = candidate_mask_for_job(x, symbol="XAUUSD", side="long", config=cfg)
    xau_short = candidate_mask_for_job(x, symbol="XAUUSD", side="short", config=cfg)
    assert eur_long.tolist()[0] is True
    assert xau_long.tolist()[2] is True
    assert xau_short.tolist()[3] is True



def test_rule_context_candidate_preset_returns_candidates() -> None:
    df = pd.DataFrame({
        "session_block_id": [3, 4, 5, 6],
        "spread_pips": [0.5, 0.5, 0.5, 0.5],
        "rsi": [50, 55, 60, 70],
        "gmma_distance": [0.0, 10.0, -20.0, 20.0],
        "gmma_distance_slope": [0.0, 1.0, -1.0, 3.0],
        "atr_regime": [2, 2, 2, 2],
        "london_expansion_vs_asia": [0.2, 0.3, 0.4, 0.5],
    })
    cfg = {
        "candidate_filter": {
            "enabled": True,
            "preset": "rule_inspired_context_v1",
            "base": {"max_spread_pips_by_symbol": {"EURUSD": 1.3}},
            "eurusd_buy_context": {},
        }
    }
    mask = candidate_mask_for_job(df, symbol="EURUSD", side="long", config=cfg)
    assert mask.sum() >= 1


def test_trading_metrics_profit_factor_and_payoff() -> None:
    from debco.trading.risk_metrics import payoff_ratio, profit_factor, summarize_trade_pnl
    import pandas as pd

    pnl = [20.0, -10.0, 20.0, -10.0]
    assert profit_factor(pnl) == 2.0
    assert payoff_ratio(pnl) == 2.0
    trades = pd.DataFrame({"pnl_pips": pnl, "pnl_R": [2.0, -1.0, 2.0, -1.0], "date": pd.date_range("2025-01-01", periods=4)})
    s = summarize_trade_pnl(trades, initial_capital=1000.0, risk_per_trade=0.02, ruin_drawdowns=[0.25], n_ruin_sims=10)
    assert s["trade_count"] == 4.0
    assert s["win_rate"] == 0.5
    assert s["profit_factor"] == 2.0
    assert s["net_R"] == 2.0


def test_fixed_threshold_and_top_percentile_masks() -> None:
    from debco.trading.threshold_policy import fixed_threshold_mask, top_percentile_mask_by_fold
    import pandas as pd

    df = pd.DataFrame({"fold": ["a", "a", "a", "b", "b"], "p": [0.1, 0.9, 0.5, 0.8, 0.2]})
    assert fixed_threshold_mask(df, probability_column="p", threshold=0.5).tolist() == [False, True, True, True, False]
    assert top_percentile_mask_by_fold(df, probability_column="p", top_percentile=50).tolist() == [False, True, True, True, False]
