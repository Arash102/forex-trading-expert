from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from debco.live.order_executor import DemoOrderExecutor
from debco.live.risk import parse_trade_profile_from_job, risk_weight_for_decision
from debco.live.signal_engine import SignalDecision
from debco.live.state_store import LiveStateStore


class FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_PLACED = 10008


class FakeMT5Client:
    def __init__(self, *, trade_mode: int = 0):
        self.mt5 = FakeMT5()
        self.trade_mode = trade_mode
        self.sent_requests: list[dict] = []

    def account_info(self):
        return SimpleNamespace(equity=1000.0, balance=1000.0, trade_mode=self.trade_mode)

    def symbol_info(self, symbol: str):
        if symbol == "EURUSD":
            return SimpleNamespace(
                digits=5,
                point=0.00001,
                trade_tick_size=0.00001,
                trade_tick_value=1.0,
                trade_tick_value_loss=1.0,
                volume_min=0.01,
                volume_max=100.0,
                volume_step=0.01,
            )
        return SimpleNamespace(
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
            trade_tick_value_loss=1.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
        )

    def symbol_info_tick(self, symbol: str):
        if symbol == "EURUSD":
            return SimpleNamespace(bid=1.09998, ask=1.10000)
        return SimpleNamespace(bid=2300.00, ask=2300.05)

    def order_send(self, request: dict):
        self.sent_requests.append(dict(request))
        return SimpleNamespace(retcode=FakeMT5.TRADE_RETCODE_DONE, order=123456, deal=654321, comment="done")


def live_spec():
    return {
        "risk_per_trade": 0.01,
        "risk_plan_weights": {
            "symbol_side_weights": {"XAUUSD|short": 0.5},
            "symbol_weights": {},
            "side_weights": {},
            "component_weights": {},
        },
    }


def decision(*, side: str = "long", symbol: str = "EURUSD", setup_id: str = "EUR_AH_ATR2_BUY") -> SignalDecision:
    profile = parse_trade_profile_from_job("EURUSD_fast_15_8_h16_long")
    return SignalDecision(
        symbol=symbol,
        timeframe="M15",
        setup_id=setup_id,
        side=side,
        magic=130103,
        signal_bar_time_utc="2026-07-06T14:45:00Z",
        decision_bar_time_utc="2026-07-06T15:00:00Z",
        action="enter",
        reason="unit_test",
        probability=0.9,
        threshold=0.5,
        dry_run=False,
        job="EURUSD_fast_15_8_h16_long",
        tp_pips=profile.tp_pips,
        sl_pips=profile.sl_pips,
        horizon_bars=profile.horizon_bars,
    )


def test_trade_profile_is_parsed_from_job_name():
    p = parse_trade_profile_from_job("XAUUSD_runner_2200_1100_h40_long")
    assert p is not None
    assert p.tp_pips == 2200
    assert p.sl_pips == 1100
    assert p.horizon_bars == 40


def test_xau_short_risk_weight_is_half():
    assert risk_weight_for_decision(live_spec(), symbol="XAUUSD", side="short", setup_id="XAU_SELL_H1DOWN_CONT") == 0.5
    assert risk_weight_for_decision(live_spec(), symbol="EURUSD", side="long", setup_id="EUR_AH_ATR2_BUY") == 1.0


def test_dry_run_builds_trade_intent_but_does_not_send_order(tmp_path: Path):
    state = LiveStateStore(tmp_path / "state.sqlite")
    client = FakeMT5Client(trade_mode=0)
    ex = DemoOrderExecutor(
        state=state,
        chart_event_dir=str(tmp_path / "events"),
        live_spec=live_spec(),
        execution_config={"dry_run": True, "enable_orders": False, "risk_per_trade": 0.01},
        mt5_client=client,
    )
    result = ex.handle_decision(decision())
    assert result is not None
    assert result["status"] == "dry_run_order_intent_created"
    assert result["volume"] == 0.12
    assert client.sent_requests == []


def test_demo_order_send_builds_market_request_with_tp_sl_and_magic(tmp_path: Path):
    state = LiveStateStore(tmp_path / "state.sqlite")
    client = FakeMT5Client(trade_mode=0)
    ex = DemoOrderExecutor(
        state=state,
        chart_event_dir=str(tmp_path / "events"),
        live_spec=live_spec(),
        execution_config={
            "dry_run": False,
            "enable_orders": True,
            "demo_only": True,
            "runtime_demo_orders_confirmed": True,
            "risk_per_trade": 0.01,
            "deviation_points": 20,
        },
        mt5_client=client,
    )
    result = ex.handle_decision(decision())
    assert result is not None
    assert result["status"] == "demo_order_sent"
    assert result["ticket"] == 123456
    req = client.sent_requests[0]
    assert req["symbol"] == "EURUSD"
    assert req["magic"] == 130103
    assert req["volume"] == 0.12
    assert req["price"] == 1.1
    assert req["sl"] == 1.0992
    assert req["tp"] == 1.1015


def test_real_account_is_blocked_even_when_orders_enabled(tmp_path: Path):
    state = LiveStateStore(tmp_path / "state.sqlite")
    client = FakeMT5Client(trade_mode=2)
    ex = DemoOrderExecutor(
        state=state,
        chart_event_dir=str(tmp_path / "events"),
        live_spec=live_spec(),
        execution_config={
            "dry_run": False,
            "enable_orders": True,
            "demo_only": True,
            "runtime_demo_orders_confirmed": True,
            "risk_per_trade": 0.01,
        },
        mt5_client=client,
    )
    result = ex.handle_decision(decision())
    assert result is not None
    assert result["status"] == "blocked_demo_only_account_check_failed"
    assert client.sent_requests == []
