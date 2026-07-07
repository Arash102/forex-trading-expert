from __future__ import annotations

import argparse

from debco.live.router import ForwardDemoRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="DEBCO v0.1.13c forward/demo live router. Safe dry-run by default.")
    parser.add_argument("--live-config", default="configs/live_router.example.json")
    parser.add_argument("--once", action="store_true", help="Poll/process once and exit. Useful for smoke checks.")
    parser.add_argument("--enable-inference", action="store_true", help="Temporarily enable live feature/model inference without editing the JSON config.")
    parser.add_argument(
        "--enable-demo-orders",
        action="store_true",
        help="Explicitly allow MT5 order_send on a verified demo account only. This also disables dry-run for this run.",
    )
    parser.add_argument(
        "--inject-test-signal",
        default=None,
        help="Dry-run by default: create one synthetic entry for the named setup_id to test chart marker/order plumbing.",
    )
    args = parser.parse_args()
    router = ForwardDemoRouter(
        args.live_config,
        inject_test_signal=args.inject_test_signal,
        force_inference_enabled=True if args.enable_inference else None,
        force_demo_orders_enabled=True if args.enable_demo_orders else None,
    )
    router.run(once=bool(args.once))


if __name__ == "__main__":
    main()
