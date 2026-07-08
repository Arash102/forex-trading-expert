from __future__ import annotations

import argparse

from debco.live.healthcheck import run_startup_healthcheck
from debco.live.router import ForwardDemoRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="DEBCO forward demo live router.")
    parser.add_argument(
        "--live-config",
        default="configs/live_router.example.json",
        help="Path to live router config JSON.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one polling cycle and exit.",
    )
    parser.add_argument(
        "--enable-inference",
        action="store_true",
        help="Enable live model inference at runtime, overriding config inference.enabled.",
    )
    parser.add_argument(
        "--enable-demo-orders",
        action="store_true",
        help="Enable real MT5 order_send only if demo-only account check passes.",
    )
    parser.add_argument(
        "--inject-test-signal",
        default=None,
        help="Create one synthetic entry for the named setup_id to test chart marker/order plumbing.",
    )
    parser.add_argument(
        "--skip-startup-healthcheck",
        action="store_true",
        help="Skip startup healthcheck before continuous runs. Not recommended for one-month demo.",
    )
    parser.add_argument(
        "--startup-healthcheck-only",
        action="store_true",
        help="Run startup healthcheck and exit without entering the router loop.",
    )

    args = parser.parse_args()

    force_inference = True if args.enable_inference else None
    force_demo_orders = True if args.enable_demo_orders else None

    should_run_healthcheck = bool(args.startup_healthcheck_only) or (
        not bool(args.once) and not bool(args.skip_startup_healthcheck)
    )

    if should_run_healthcheck:
        hc = run_startup_healthcheck(
            args.live_config,
            force_inference_enabled=force_inference,
            force_demo_orders_enabled=force_demo_orders,
            print_report=True,
        )
        if not hc.ready:
            raise SystemExit(2)
        if args.startup_healthcheck_only:
            return

    router = ForwardDemoRouter(
        args.live_config,
        force_inference_enabled=force_inference,
        force_demo_orders_enabled=force_demo_orders,
        inject_test_signal=args.inject_test_signal,
    )
    router.run(once=bool(args.once))


if __name__ == "__main__":
    main()
