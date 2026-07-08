from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak_before_mt5_repair_{stamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def latest_live_config_backup(live_path: Path) -> Path | None:
    pattern = live_path.name + ".bak_*"
    candidates = sorted(live_path.parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def has_complete_explicit_login(mt5_cfg: Mapping[str, Any]) -> bool:
    return bool(mt5_cfg.get("login")) and bool(mt5_cfg.get("password")) and bool(mt5_cfg.get("server"))


def sanitized_mt5(mt5_cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": mt5_cfg.get("enabled"),
        "terminal_path_set": bool(mt5_cfg.get("terminal_path")),
        "login": mt5_cfg.get("login"),
        "password_set": bool(mt5_cfg.get("password")),
        "server": mt5_cfg.get("server"),
        "history_bars": mt5_cfg.get("history_bars"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair live_router.local.json MT5 connection after demo-month config preparation."
    )
    parser.add_argument("--live-config", default="configs/live_router.local.json")
    parser.add_argument("--from-latest-backup", action="store_true", help="Restore the mt5 block from the newest live_router.local.json.bak_* file if available.")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    live_path = Path(args.live_config)
    cfg = read_json(live_path)
    mt5 = dict(cfg.get("mt5", {}) or {})

    before = sanitized_mt5(mt5)
    source = "current_config_cleanup"

    if args.from_latest_backup:
        bak = latest_live_config_backup(live_path)
        if bak is not None:
            bak_cfg = read_json(bak)
            bak_mt5 = dict(bak_cfg.get("mt5", {}) or {})
            if bak_mt5:
                # Restore the known-working connection fields, but keep launch history_bars if already set.
                history_bars = mt5.get("history_bars", bak_mt5.get("history_bars", 5000))
                mt5 = bak_mt5
                mt5["history_bars"] = history_bars
                source = f"backup:{bak}"

    # Main repair: incomplete explicit login is dangerous for mt5.initialize.
    # If login exists without both password and server, remove login and let MT5 use the already-open terminal session.
    if mt5.get("login") and not has_complete_explicit_login(mt5):
        mt5.pop("login", None)

    mt5["enabled"] = True
    mt5["history_bars"] = int(mt5.get("history_bars", 5000))

    cfg["mt5"] = mt5

    if not args.no_backup:
        print(f"backup: {backup(live_path)}")

    write_json(live_path, cfg)

    print("MT5 connection config repaired.")
    print(f"source: {source}")
    print("before:", json.dumps(before, ensure_ascii=False))
    print("after: ", json.dumps(sanitized_mt5(mt5), ensure_ascii=False))
    print("Next: python scripts/17_diagnose_live_features.py --live-config configs/live_router.local.json")


if __name__ == "__main__":
    main()
