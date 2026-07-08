from __future__ import annotations

import json
from pathlib import Path

from debco.live.config import DEFAULT_SETUP_MAGIC_NUMBERS
from debco.live.demo_launch_lock import validate_static_launch_lock


def _core_spec() -> dict:
    rows = []
    for setup_id in DEFAULT_SETUP_MAGIC_NUMBERS:
        rows.append(
            {
                "setup_id": setup_id,
                "symbol": "EURUSD" if setup_id.startswith("EUR") else "XAUUSD",
                "side": "short" if ("SELL" in setup_id or "SHORT" in setup_id) else "long",
                "policy": "top_percentile",
                "threshold": None,
                "top_percentile": 2.0,
            }
        )
    return {"selected_setups": rows, "risk_per_trade": 0.01}


def _live_cfg(tmp_path: Path) -> dict:
    models_dir = tmp_path / "data" / "live_models"
    for setup_id in DEFAULT_SETUP_MAGIC_NUMBERS:
        d = models_dir / setup_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "artifact.json").write_text("{}", encoding="utf-8")

    spec_path = tmp_path / "data" / "final_strategy_report" / "live_execution_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(_core_spec()), encoding="utf-8")

    return {
        "router_id": "debco_forward_demo_v0_1_14a",
        "mode": "demo",
        "live_execution_spec_path": str(spec_path),
        "state_db_path": str(tmp_path / "data" / "live_state" / "forward_demo.sqlite"),
        "chart_event_dir": str(tmp_path / "data" / "live_state" / "chart_events"),
        "symbols": ["EURUSD", "XAUUSD"],
        "timeframe": "M15",
        "mt5": {"enabled": True, "login": 123456, "server": "Broker-Demo"},
        "execution": {
            "dry_run": False,
            "enable_orders": True,
            "risk_per_trade": 0.01,
            "horizon_exit_enabled": True,
            "one_signal_per_setup_per_bar": True,
            "demo_only": True,
            "require_demo_orders_cli_flag": True,
        },
        "safety": {
            "max_orders_per_bar_per_setup": 1,
            "max_open_trades": 8,
            "max_trades_per_day": 8,
            "max_trades_per_symbol_per_day": 4,
            "stop_after_daily_losses": 2,
            "max_open_trades_per_symbol": 4,
            "max_open_trades_per_setup": 1,
            "block_opposite_symbol_positions": True,
        },
        "setup_magic_numbers": dict(DEFAULT_SETUP_MAGIC_NUMBERS),
        "chart_markers": {
            "enabled": True,
            "screenshot_on_entry": True,
            "screenshot_on_exit": True,
        },
        "inference": {
            "enabled": True,
            "live_models_dir": str(models_dir),
            "require_live_models_for_all_setups": True,
        },
        "position_manager": {
            "enabled": True,
            "sync_open_positions": True,
            "horizon_exit_enabled": True,
        },
        "reports": {
            "enabled": True,
            "output_dir": str(tmp_path / "data" / "live_reports"),
            "write_on_each_new_bar": True,
        },
        "runtime": {
            "max_consecutive_errors": 10,
            "reconnect_on_error": True,
        },
    }


def _launch_lock() -> dict:
    return {
        "required": {
            "order_mode": "demo_orders",
            "symbols": ["EURUSD", "XAUUSD"],
            "timeframe": "M15",
            "setup_count": 12,
            "require_core_12_setup_ids": True,
            "require_live_models_for_all_setups": True,
            "min_live_model_artifacts": 12,
            "require_entry_exit_screenshots": True,
            "demo_run_min_days": 30,
        },
        "limits": {
            "max_risk_per_trade": 0.01,
            "max_open_trades": 8,
            "max_trades_per_day": 8,
            "max_trades_per_symbol_per_day": 4,
            "stop_after_daily_losses": 2,
            "max_open_trades_per_symbol": 4,
            "max_open_trades_per_setup": 1,
            "max_orders_per_bar_per_setup": 1,
            "max_consecutive_errors": 10,
        },
        "account": {"server_must_contain_any": ["demo"], "allowed_logins": []},
        "acknowledgements": {
            "strategy_frozen": True,
            "features_frozen": True,
            "models_frozen": True,
            "demo_account_verified": True,
            "manual_preflight_done": True,
            "risk_acceptance": True,
        },
    }


def test_valid_demo_month_launch_lock_passes(tmp_path: Path):
    live_cfg = _live_cfg(tmp_path)
    lock = _launch_lock()
    assert validate_static_launch_lock(live_cfg, lock, _core_spec(), repo_root=tmp_path) == []


def test_blocks_non_demo_execution(tmp_path: Path):
    live_cfg = _live_cfg(tmp_path)
    live_cfg["execution"]["demo_only"] = False
    issues = validate_static_launch_lock(live_cfg, _launch_lock(), _core_spec(), repo_root=tmp_path)
    assert any("demo_only" in issue for issue in issues)


def test_blocks_risk_above_launch_limit(tmp_path: Path):
    live_cfg = _live_cfg(tmp_path)
    live_cfg["execution"]["risk_per_trade"] = 0.02
    issues = validate_static_launch_lock(live_cfg, _launch_lock(), _core_spec(), repo_root=tmp_path)
    assert any("risk_per_trade" in issue and "<=" in issue for issue in issues)


def test_requires_manual_acknowledgements(tmp_path: Path):
    live_cfg = _live_cfg(tmp_path)
    lock = _launch_lock()
    lock["acknowledgements"]["manual_preflight_done"] = False
    issues = validate_static_launch_lock(live_cfg, lock, _core_spec(), repo_root=tmp_path)
    assert any("manual_preflight_done" in issue for issue in issues)


def test_blocks_missing_live_model_artifacts(tmp_path: Path):
    live_cfg = _live_cfg(tmp_path)
    # Remove one artifact to simulate incomplete live model registry.
    first = next(iter(DEFAULT_SETUP_MAGIC_NUMBERS))
    Path(live_cfg["inference"]["live_models_dir"]).joinpath(first, "artifact.json").unlink()
    issues = validate_static_launch_lock(live_cfg, _launch_lock(), _core_spec(), repo_root=tmp_path)
    assert any("live model artifacts" in issue for issue in issues)
