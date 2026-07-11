from __future__ import annotations

import argparse
from pathlib import Path

from debco.live.position_history_sync import enrich_closed_positions_from_history
from debco.live.reporting import write_daily_report
from debco.live.router import ForwardDemoRouter


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync MT5 positions into state/report without sending new orders.")
    ap.add_argument("--live-config", default="configs/live_router.local.json")
    ap.add_argument("--day-utc", default=None)
    ap.add_argument("--history-lookback-days", type=int, default=14)
    args = ap.parse_args()

    router = ForwardDemoRouter(args.live_config)
    client = router.connect_mt5()
    try:
        if router.position_manager is None:
            raise RuntimeError("position_manager was not initialized after MT5 connection")
        sync_result = router.position_manager.sync_open_positions_from_mt5()
        history_result = enrich_closed_positions_from_history(
            router.state,
            client,
            lookback_days=args.history_lookback_days,
        )
        report_dir = str((router.cfg.get("reports", {}) or {}).get("output_dir", "data/live_reports"))
        report = write_daily_report(router.state, report_dir, day_utc=args.day_utc)
        print("DEBCO report-only position sync")
        print("================================")
        print(f"state_db: {router.paths.state_db_path}")
        print(f"report_dir: {Path(report_dir)}")
        print(f"open_sync: {sync_result}")
        print(f"history_sync: {history_result.to_payload()}")
        print(f"report: {report}")
        print("OK: report-only sync completed. No orders were sent.")
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
