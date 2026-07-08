from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

from debco.live.config import DEFAULT_SETUP_MAGIC_NUMBERS, validate_router_bundle


EXPECTED_CORE_SETUP_IDS = tuple(DEFAULT_SETUP_MAGIC_NUMBERS.keys())


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing JSON file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out


def _norm_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, (list, tuple, set)):
        return [str(x) for x in values]
    return []


def _read_nested(data: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _path_from_root(repo_root: str | Path, raw_path: Any) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return Path(repo_root) / path


def selected_setup_ids(spec: Mapping[str, Any]) -> list[str]:
    rows = spec.get("selected_setups", []) or []
    setup_ids: list[str] = []
    for row in rows:
        setup_id = str(row.get("setup_id", "")).strip()
        if setup_id:
            setup_ids.append(setup_id)
    return setup_ids


def count_live_model_artifacts(models_dir: str | Path) -> int:
    p = Path(models_dir)
    if not p.exists():
        return 0
    return len(list(p.glob("*/artifact.json")))


def run_git_command(repo_root: str | Path, args: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as exc:
        return 127, str(exc)
    return proc.returncode, proc.stdout.strip()


def validate_git_lock(launch_lock: Mapping[str, Any], repo_root: str | Path = ".") -> list[str]:
    git_cfg = launch_lock.get("git", {}) or {}
    issues: list[str] = []

    if not _as_bool(git_cfg.get("enabled", True), True):
        return issues

    rc, inside = run_git_command(repo_root, ["rev-parse", "--is-inside-work-tree"])
    if rc != 0 or inside.strip().lower() != "true":
        return [f"git lock failed: not inside a git worktree at {repo_root}"]

    if _as_bool(git_cfg.get("require_clean_worktree", True), True):
        rc, status = run_git_command(repo_root, ["status", "--porcelain"])
        if rc != 0:
            issues.append(f"git status failed: {status}")
        elif status.strip():
            issues.append("git worktree must be clean before demo launch")

    allowed_branches = _norm_list(git_cfg.get("allowed_branches", ["main"]))
    if allowed_branches:
        rc, branch = run_git_command(repo_root, ["branch", "--show-current"])
        if rc != 0:
            issues.append(f"git branch check failed: {branch}")
        elif branch not in allowed_branches:
            issues.append(f"current git branch must be one of {allowed_branches}, got {branch!r}")

    required_ref = str(git_cfg.get("required_ref", "")).strip()
    if required_ref:
        rc_head, head = run_git_command(repo_root, ["rev-parse", "HEAD"])
        rc_ref, ref = run_git_command(repo_root, ["rev-parse", required_ref])
        if rc_ref != 0:
            issues.append(f"required git ref not found: {required_ref}")
        elif rc_head != 0:
            issues.append(f"git HEAD check failed: {head}")
        elif head != ref:
            issues.append(f"HEAD must match required_ref={required_ref}; HEAD={head[:12]}, ref={ref[:12]}")

    return issues


def validate_manual_acknowledgements(launch_lock: Mapping[str, Any]) -> list[str]:
    ack = launch_lock.get("acknowledgements", {}) or {}
    required = [
        "strategy_frozen",
        "features_frozen",
        "models_frozen",
        "demo_account_verified",
        "manual_preflight_done",
        "risk_acceptance",
    ]
    issues: list[str] = []
    for key in required:
        if not _as_bool(ack.get(key), False):
            issues.append(f"acknowledgements.{key} must be true")
    return issues


def validate_static_launch_lock(
    live_cfg: Mapping[str, Any],
    launch_lock: Mapping[str, Any],
    spec: Mapping[str, Any],
    repo_root: str | Path = ".",
) -> list[str]:
    issues: list[str] = []
    limits = launch_lock.get("limits", {}) or {}
    required = launch_lock.get("required", {}) or {}
    account = launch_lock.get("account", {}) or {}

    issues.extend(validate_router_bundle(live_cfg, spec))
    issues.extend(validate_manual_acknowledgements(launch_lock))

    symbols = [str(x).upper() for x in live_cfg.get("symbols", [])]
    expected_symbols = [str(x).upper() for x in _norm_list(required.get("symbols", ["EURUSD", "XAUUSD"]))]
    if sorted(symbols) != sorted(expected_symbols):
        issues.append(f"live config symbols must be exactly {expected_symbols}, got {symbols}")

    timeframe = str(live_cfg.get("timeframe", "")).upper()
    expected_timeframe = str(required.get("timeframe", "M15")).upper()
    if timeframe != expected_timeframe:
        issues.append(f"timeframe must be {expected_timeframe}, got {timeframe!r}")

    setup_ids = selected_setup_ids(spec)
    expected_setup_count = _as_int(required.get("setup_count"), len(EXPECTED_CORE_SETUP_IDS))
    if expected_setup_count is not None and len(setup_ids) != expected_setup_count:
        issues.append(f"selected setup count must be {expected_setup_count}, got {len(setup_ids)}")

    if _as_bool(required.get("require_core_12_setup_ids", True), True):
        missing = sorted(set(EXPECTED_CORE_SETUP_IDS).difference(set(setup_ids)))
        extra = sorted(set(setup_ids).difference(set(EXPECTED_CORE_SETUP_IDS)))
        if missing:
            issues.append(f"live execution spec missing core setup ids: {missing}")
        if extra:
            issues.append(f"live execution spec contains unexpected setup ids: {extra}")

    execution = live_cfg.get("execution", {}) or {}
    expected_order_mode = str(required.get("order_mode", "demo_orders")).lower()
    dry_run = _as_bool(execution.get("dry_run", True), True)
    enable_orders = _as_bool(execution.get("enable_orders", False), False)
    demo_only = _as_bool(execution.get("demo_only", True), True)
    require_cli_flag = _as_bool(execution.get("require_demo_orders_cli_flag", True), True)

    if expected_order_mode == "demo_orders":
        if dry_run:
            issues.append("execution.dry_run must be false for one-month demo order test")
        if not enable_orders:
            issues.append("execution.enable_orders must be true for one-month demo order test")
    elif expected_order_mode == "dry_run":
        if not dry_run:
            issues.append("execution.dry_run must be true for dry-run launch lock")
        if enable_orders:
            issues.append("execution.enable_orders must be false for dry-run launch lock")
    else:
        issues.append(f"unsupported required.order_mode={expected_order_mode!r}")

    if not demo_only:
        issues.append("execution.demo_only must be true")
    if not require_cli_flag:
        issues.append("execution.require_demo_orders_cli_flag must be true")
    if not _as_bool(execution.get("horizon_exit_enabled", True), True):
        issues.append("execution.horizon_exit_enabled must be true")
    if not _as_bool(execution.get("one_signal_per_setup_per_bar", True), True):
        issues.append("execution.one_signal_per_setup_per_bar must be true")

    risk = _as_float(execution.get("risk_per_trade"), None)
    max_risk = _as_float(limits.get("max_risk_per_trade"), 0.01)
    min_risk = _as_float(limits.get("min_risk_per_trade"), 0.0)
    if risk is None:
        issues.append("execution.risk_per_trade must be a finite number")
    else:
        if min_risk is not None and risk <= min_risk:
            issues.append(f"execution.risk_per_trade must be > {min_risk}, got {risk}")
        if max_risk is not None and risk > max_risk:
            issues.append(f"execution.risk_per_trade must be <= {max_risk}, got {risk}")

    safety = live_cfg.get("safety", {}) or {}
    integer_limits = {
        "max_open_trades": 8,
        "max_trades_per_day": 8,
        "max_trades_per_symbol_per_day": 4,
        "stop_after_daily_losses": 2,
        "max_open_trades_per_symbol": 4,
        "max_open_trades_per_setup": 1,
        "max_orders_per_bar_per_setup": 1,
    }
    for key, default_limit in integer_limits.items():
        limit_value = _as_int(limits.get(key), default_limit)
        cfg_value = _as_int(safety.get(key), None)
        if cfg_value is None:
            issues.append(f"safety.{key} must be configured")
            continue
        if limit_value is not None and cfg_value > limit_value:
            issues.append(f"safety.{key} must be <= {limit_value}, got {cfg_value}")

    if not _as_bool(safety.get("block_opposite_symbol_positions", True), True):
        issues.append("safety.block_opposite_symbol_positions must be true")

    inference = live_cfg.get("inference", {}) or {}
    if not _as_bool(inference.get("enabled", False), False):
        issues.append("inference.enabled must be true")
    if _as_bool(required.get("require_live_models_for_all_setups", True), True):
        if not _as_bool(inference.get("require_live_models_for_all_setups", False), False):
            issues.append("inference.require_live_models_for_all_setups must be true")

    live_models_dir = _path_from_root(repo_root, inference.get("live_models_dir", "data/live_models"))
    if not live_models_dir.exists():
        issues.append(f"missing live models dir: {live_models_dir}")
    else:
        artifact_count = count_live_model_artifacts(live_models_dir)
        min_artifacts = _as_int(required.get("min_live_model_artifacts"), expected_setup_count or 12)
        if min_artifacts is not None and artifact_count < min_artifacts:
            issues.append(f"expected at least {min_artifacts} live model artifacts, found {artifact_count}")

    reports = live_cfg.get("reports", {}) or {}
    if not _as_bool(reports.get("enabled", True), True):
        issues.append("reports.enabled must be true")
    if not _as_bool(reports.get("write_on_each_new_bar", True), True):
        issues.append("reports.write_on_each_new_bar must be true")
    if not str(reports.get("output_dir", "")).strip():
        issues.append("reports.output_dir must be configured")

    pm = live_cfg.get("position_manager", {}) or {}
    if not _as_bool(pm.get("enabled", True), True):
        issues.append("position_manager.enabled must be true")
    if not _as_bool(pm.get("sync_open_positions", True), True):
        issues.append("position_manager.sync_open_positions must be true")
    if not _as_bool(pm.get("horizon_exit_enabled", True), True):
        issues.append("position_manager.horizon_exit_enabled must be true")

    chart_markers = live_cfg.get("chart_markers", {}) or {}
    if not _as_bool(chart_markers.get("enabled", True), True):
        issues.append("chart_markers.enabled must be true")
    if _as_bool(required.get("require_entry_exit_screenshots", True), True):
        if not _as_bool(chart_markers.get("screenshot_on_entry", False), False):
            issues.append("chart_markers.screenshot_on_entry must be true")
        if not _as_bool(chart_markers.get("screenshot_on_exit", False), False):
            issues.append("chart_markers.screenshot_on_exit must be true")

    runtime = live_cfg.get("runtime", {}) or {}
    if not _as_bool(runtime.get("reconnect_on_error", True), True):
        issues.append("runtime.reconnect_on_error must be true")
    max_errors = _as_int(runtime.get("max_consecutive_errors"), None)
    max_allowed_errors = _as_int(limits.get("max_consecutive_errors"), 10)
    if max_errors is None:
        issues.append("runtime.max_consecutive_errors must be configured")
    elif max_allowed_errors is not None and max_errors > max_allowed_errors:
        issues.append(f"runtime.max_consecutive_errors must be <= {max_allowed_errors}, got {max_errors}")

    mt5 = live_cfg.get("mt5", {}) or {}
    if not _as_bool(mt5.get("enabled", True), True):
        issues.append("mt5.enabled must be true")

    expected_server_keywords = [x.lower() for x in _norm_list(account.get("server_must_contain_any", []))]
    server = str(mt5.get("server", "") or "").lower()
    if expected_server_keywords and not any(k in server for k in expected_server_keywords):
        issues.append(f"mt5.server must contain one of {expected_server_keywords}; got {mt5.get('server')!r}")

    allowed_logins = _norm_list(account.get("allowed_logins", []))
    login = str(mt5.get("login", "") or "")
    # Terminal-session mode:
    # If allowed_logins is empty, do not require mt5.login inside live_router.local.json.
    # The real runtime account is verified by the startup healthcheck output.
    if allowed_logins:
        if not login:
            issues.append("mt5.login is required when launch_lock.account.allowed_logins is not empty")
        elif login not in allowed_logins:
            issues.append("mt5.login is not in launch_lock.account.allowed_logins")

    live_execution_spec_path = _path_from_root(repo_root, live_cfg.get("live_execution_spec_path", ""))
    if not live_execution_spec_path.exists():
        issues.append(f"missing live execution spec: {live_execution_spec_path}")

    state_db_path = str(live_cfg.get("state_db_path", "")).strip()
    if not state_db_path:
        issues.append("state_db_path must be configured")

    min_days = _as_int(required.get("demo_run_min_days"), 30)
    if min_days is None or min_days < 30:
        issues.append("required.demo_run_min_days must be at least 30")

    return issues


def validate_demo_month_launch_lock(
    live_config_path: str | Path,
    launch_lock_path: str | Path,
    repo_root: str | Path = ".",
    check_git: bool = True,
) -> list[str]:
    live_cfg = load_json(live_config_path)
    launch_lock = load_json(launch_lock_path)
    spec_path = _path_from_root(repo_root, live_cfg.get("live_execution_spec_path", "data/final_strategy_report/live_execution_spec.json"))
    spec = load_json(spec_path)

    issues = validate_static_launch_lock(live_cfg, launch_lock, spec, repo_root=repo_root)
    if check_git:
        issues.extend(validate_git_lock(launch_lock, repo_root=repo_root))
    return issues
