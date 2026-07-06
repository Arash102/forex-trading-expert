from __future__ import annotations

import argparse

from debco.live.router import ForwardDemoRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="DEBCO v0.1.13a forward/demo live router. Starts dry-run by default.")
    parser.add_argument("--live-config", default="configs/live_router.example.json")
    parser.add_argument("--once", action="store_true", help="Poll/process once and exit. Useful for smoke checks.")
    parser.add_argument(
        "--inject-test-signal",
        default=None,
        help="Dry-run only: create one synthetic entry for the named setup_id to test chart marker/screenshot plumbing.",
    )
    args = parser.parse_args()
    router = ForwardDemoRouter(args.live_config, inject_test_signal=args.inject_test_signal)
    router.run(once=bool(args.once))


if __name__ == "__main__":
    main()
