from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Callable

from debco.live import feature_diagnostics as fd


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sanitized_mt5(cfg: dict[str, Any]) -> dict[str, Any]:
    mt5 = cfg.get("mt5", {}) or {}
    return {
        "enabled": mt5.get("enabled"),
        "terminal_path_set": bool(mt5.get("terminal_path")),
        "login": mt5.get("login"),
        "password_set": bool(mt5.get("password")),
        "server": mt5.get("server"),
        "history_bars": mt5.get("history_bars"),
    }


def _formatter() -> Callable[..., str]:
    if hasattr(fd, "format_feature_diagnostics"):
        return fd.format_feature_diagnostics
    if hasattr(fd, "format_feature_diagnostic_report"):
        return fd.format_feature_diagnostic_report
    raise ImportError(
        "debco.live.feature_diagnostics has no formatter. Expected "
        "format_feature_diagnostics or format_feature_diagnostic_report."
    )


def _diagnose(*, live_config: str, max_bars: int, max_examples: int | None = None) -> dict[str, Any]:
    fn = fd.diagnose_live_feature_state
    sig = inspect.signature(fn)
    kwargs: dict[str, Any] = {}

    if "live_config_path" in sig.parameters:
        kwargs["live_config_path"] = live_config
    else:
        kwargs["live_config"] = live_config

    if "force_inference_enabled" in sig.parameters:
        kwargs["force_inference_enabled"] = True
    if "force_demo_orders_enabled" in sig.parameters:
        kwargs["force_demo_orders_enabled"] = True
    if "max_bars" in sig.parameters:
        kwargs["max_bars"] = max_bars
    elif "max_examples" in sig.parameters:
        kwargs["max_examples"] = max_examples if max_examples is not None else 12

    return fn(**kwargs)


def _write_json_report(report: dict[str, Any], path: str | Path) -> None:
    if hasattr(fd, "dump_feature_diagnostics_json"):
        fd.dump_feature_diagnostics_json(report, Path(path))
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose live feature validity before one-month demo launch."
    )
    parser.add_argument("--live-config", default="configs/live_router.local.json")
    parser.add_argument("--max-bars", type=int, default=5000)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Backward-compatible option for older diagnostic implementations.",
    )
    parser.add_argument(
        "--all-setups",
        action="store_true",
        help="Show invalid features for all setups, not only candidate-passed setups.",
    )
    parser.add_argument(
        "--json-output",
        default="data/live_diagnostics/feature_diagnostics_latest.json",
    )
    args = parser.parse_args()

    try:
        report = _diagnose(
            live_config=args.live_config,
            max_bars=args.max_bars,
            max_examples=args.max_examples,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "mt5.initialize failed" in msg:
            cfg = _read_json(args.live_config)
            print("DEBCO LIVE FEATURE DIAGNOSTICS")
            print("MT5 CONNECTION ERROR")
            print(f"error: {msg}")
            print("mt5_config:", json.dumps(_sanitized_mt5(cfg), ensure_ascii=False))
            print("")
            print("FIX:")
            print("python scripts/16c_repair_demo_month_mt5_connection.py --live-config configs/live_router.local.json --from-latest-backup")
            raise SystemExit(2) from exc
        raise

    fmt = _formatter()
    try:
        text = fmt(report, only_candidate_pass=not args.all_setups)
    except TypeError:
        text = fmt(report)
    print(text)

    if args.json_output:
        _write_json_report(report, args.json_output)
        print(f"\njson_report: {Path(args.json_output)}")

    summary = report.get("summary") or {}
    candidate_problem_count = int(summary.get("candidate_pass_problem_count", 0) or 0)
    if candidate_problem_count > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
