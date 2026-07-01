from __future__ import annotations


def main() -> None:
    """Future live router.

    Rules:
    - Python calculates all features/signals.
    - Only last closed candle is evaluated.
    - Orders are sent via MetaTrader5 Python API.
    - MQL monitor must not calculate signals.
    """
    raise NotImplementedError("Live execution will be implemented after research/backtest modules are locked.")


if __name__ == "__main__":
    main()
