from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from debco.live.position_manager import LivePositionManager
from debco.live.state_store import LiveStateStore


class FakeMT5:
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_PLACED = 10008


class FakeMT5Client:
    def __init__(self):
        self.mt5 = FakeMT5()
        self.sent_requests = []
        self.positions = [
            SimpleNamespace(
                ticket=111,
                symbol="EURUSD",
                magic=130103,
                type=FakeMT5.POSITION_TYPE_BUY,
                volume=0.1,
                price_open=1.1,
                sl=1.099,
                tp=1.102,
                time=1760000000,
                profit=0.0,
            )
        ]

    def positions_get(self, **kwargs):
        return list(self.positions)

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=1.101, ask=1.1012)

    def order_send(self, request):
        self.sent_requests.append(dict(request))
        return SimpleNamespace(retcode=FakeMT5.TRADE_RETCODE_DONE, order=222, deal=333)


def test_position_manager_syncs_mt5_positions(tmp_path: Path):
    state = LiveStateStore(tmp_path / "state.sqlite")
    pm = LivePositionManager(
        state=state,
        mt5_client=FakeMT5Client(),
        setup_magic_numbers={"EUR_AH_ATR2_BUY": 130103},
        execution_config={},
    )
    result = pm.sync_open_positions_from_mt5()
    assert result["synced_positions"] == 1
    rows = state.list_open_positions()
    assert len(rows) == 1
    assert rows[0]["setup_id"] == "EUR_AH_ATR2_BUY"
    assert rows[0]["mt5_position_ticket"] == "111"


def test_horizon_exit_sends_close_order_when_due(tmp_path: Path):
    state = LiveStateStore(tmp_path / "state.sqlite")
    client = FakeMT5Client()
    state.upsert_position(
        {
            "mt5_position_ticket": "111",
            "symbol": "EURUSD",
            "setup_id": "EUR_AH_ATR2_BUY",
            "side": "long",
            "magic": 130103,
            "volume": 0.1,
            "entry_price": 1.1,
            "signal_bar_time_utc": "2026-01-01T00:00:00Z",
            "decision_bar_time_utc": "2026-01-01T00:15:00Z",
            "horizon_bars": 2,
            "status": "open",
        }
    )
    pm = LivePositionManager(
        state=state,
        mt5_client=client,
        setup_magic_numbers={"EUR_AH_ATR2_BUY": 130103},
        execution_config={"enable_orders": True, "dry_run": False, "runtime_demo_orders_confirmed": True},
        chart_event_dir=str(tmp_path / "events"),
    )
    result = pm.manage_horizon_exits(timeframe="M15", current_bar_times={"EURUSD": "2026-01-01T00:45:00Z"})
    assert result["horizon_exit_sent"] == 1
    assert client.sent_requests[0]["position"] == 111
    rows = state.list_open_positions()
    assert rows == []
