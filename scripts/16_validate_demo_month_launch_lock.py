from __future__ import annotations

import argparse
from pathlib import Path

from debco.live.demo_launch_lock import validate_demo_month_launch_lock


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the final one-month demo launch lock before enabling unattended demo execution."
    )
    parser.add_argument("--live-config", default="configs/live_router.local.json")
    parser.add_argument("--launch-lock", default="configs/demo_month_launch_lock.local.json")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--skip-git-check", action="store_true")
    args = parser.parse_args()

    print("DEBCO one-month demo launch lock")
    print(f"live_config:  {Path(args.live_config)}")
    print(f"launch_lock:  {Path(args.launch_lock)}")

    issues = validate_demo_month_launch_lock(
        live_config_path=args.live_config,
        launch_lock_path=args.launch_lock,
        repo_root=args.repo_root,
        check_git=not args.skip_git_check,
    )

    if issues:
        print("NOT LOCKED")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("LOCKED: one-month demo launch checks passed.")
    print("Next: run a short supervised session, then start the router with --enable-inference --enable-demo-orders.")


if __name__ == "__main__":
    main()
