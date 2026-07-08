from __future__ import annotations

import argparse
import copy
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


DEFAULT_LAUNCH_LOCK: dict[str, Any] = {
    "launch_id": "debco_demo_month_v0_1_14c",
    "purpose": "Final local launch gate for one-month MT5 demo account execution only.",
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
        "min_risk_per_trade": 0.0,
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
    "account": {
        "expected_account_type": "demo",
        "server_must_contain_any": [],
        "allowed_logins": ["878713"],
    },
    "git": {
        "enabled": True,
        "require_clean_worktree": True,
        "allowed_branches": ["main"],
        "required_ref": "",
    },
    "acknowledgements": {
        "strategy_frozen": True,
        "features_frozen": True,
        "models_frozen": True,
        "demo_account_verified": True,
        "manual_preflight_done": True,
        "risk_acceptance": True,
    },
    "notes": [
        "Prepared by scripts/16b_prepare_demo_month_local_configs.py.",
        "MT5 login is whitelisted in launch_lock. The live MT5 connection block is not forced to use login/password/server unless explicitly requested.",
    ],
}


# Important safety note:
# Do NOT force mt5.login into live_router.local.json by default.
# The user's previous working setup may rely on an already-open/logged-in MT5 terminal.
# MetaTrader5.initialize(login=...) without matching password/server may fail with Invalid params.
LIVE_ROUTER_OVERRIDES: dict[str, Any] = {
    "mode": "demo",
    "symbols": ["EURUSD", "XAUUSD"],
    "timeframe": "M15",
    "broker_time": {
        "server_utc_offset_hours": 2,
        "timestamps_note": "MT5 bar timestamps are treated as broker-server time for training/live consistency. Broker server timezone observed as UTC+2.",
    },
    "mt5": {
        "enabled": True,
        "history_bars": 5000,
    },
    "dxy": {
        "enabled": True,
        "fail_if_missing": True,
    },
    "execution": {
        "dry_run": False,
        "enable_orders": True,
        "demo_only": True,
        "require_demo_orders_cli_flag": True,
        "runtime_demo_orders_confirmed": False,
        "risk_per_trade": 0.01,
        "horizon_exit_enabled": True,
        "one_signal_per_setup_per_bar": True,
    },
    "safety": {
        "max_open_trades": 8,
        "max_trades_per_day": 8,
        "max_trades_per_symbol_per_day": 4,
        "stop_after_daily_losses": 2,
        "max_open_trades_per_symbol": 4,
        "max_open_trades_per_setup": 1,
        "max_orders_per_bar_per_setup": 1,
        "block_opposite_symbol_positions": True,
    },
    "inference": {
        "enabled": True,
        "ml_config_path": "configs/ml_config.local.json",
        "feature_config_path": "configs/features_config.local.json",
        "live_models_dir": "data/live_models",
        "require_live_models_for_all_setups": True,
    },
    "position_manager": {
        "enabled": True,
        "sync_open_positions": True,
        "horizon_exit_enabled": True,
    },
    "chart_markers": {
        "enabled": True,
        "screenshot_on_entry": True,
        "screenshot_on_exit": True,
    },
    "reports": {
        "enabled": True,
        "write_on_each_new_bar": True,
        "output_dir": "data/live_reports",
    },
    "runtime": {
        "max_consecutive_errors": 10,
        "retry_sleep_seconds": 10,
        "reconnect_on_error": True,
    },
    "logging": {
        "print_heartbeat": True,
    },
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = copy.deepcopy(value)
    return out


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak_{stamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def _has_complete_explicit_mt5_login(mt5_cfg: Mapping[str, Any]) -> bool:
    return bool(mt5_cfg.get("login")) and bool(mt5_cfg.get("password")) and bool(mt5_cfg.get("server"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare local configs for the one-month DEBCO demo launch without breaking the MT5 connection block."
    )
    parser.add_argument("--live-config", default="configs/live_router.local.json")
    parser.add_argument("--launch-lock", default="configs/demo_month_launch_lock.local.json")
    parser.add_argument("--login", default="878713", help="Demo account login used for launch-lock whitelist, not forced into mt5 config by default.")
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--write-mt5-login", action="store_true", help="Only use this if live_router.local.json already has mt5.password and mt5.server, or you pass them manually later.")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    live_path = Path(args.live_config)
    lock_path = Path(args.launch_lock)

    live_cfg = read_json(live_path)
    original_mt5 = copy.deepcopy(live_cfg.get("mt5", {}) or {})

    overrides = copy.deepcopy(LIVE_ROUTER_OVERRIDES)
    overrides["execution"]["risk_per_trade"] = float(args.risk)

    if args.write_mt5_login:
        overrides["mt5"]["login"] = int(args.login) if str(args.login).isdigit() else args.login

    prepared_live_cfg = deep_merge(live_cfg, overrides)

    # If login was previously injected but password/server are absent, remove it and rely on the already-open terminal session.
    mt5_after = prepared_live_cfg.setdefault("mt5", {})
    if not args.write_mt5_login and not _has_complete_explicit_mt5_login(original_mt5):
        for key in ["login", "password", "server"]:
            if key in original_mt5:
                mt5_after[key] = original_mt5[key]
        # Remove incomplete explicit login because mt5.initialize(login=...) may fail without password/server.
        if mt5_after.get("login") and not (mt5_after.get("password") and mt5_after.get("server")):
            mt5_after.pop("login", None)

    prepared_lock = copy.deepcopy(DEFAULT_LAUNCH_LOCK)
    prepared_lock["account"]["allowed_logins"] = [str(args.login)]
    prepared_lock["limits"]["max_risk_per_trade"] = float(args.risk)

    if not args.no_backup:
        for p in [live_path, lock_path]:
            bak = backup(p)
            if bak is not None:
                print(f"backup: {bak}")

    write_json(live_path, prepared_live_cfg)
    write_json(lock_path, prepared_lock)

    print("Prepared local demo-month configs:")
    print(f"- {live_path}")
    print(f"- {lock_path}")
    print("MT5 connection mode: preserved terminal/session style unless --write-mt5-login was explicitly used.")
    print("Next: run scripts/17_diagnose_live_features.py, then the startup healthcheck.")


if __name__ == "__main__":
    main()
