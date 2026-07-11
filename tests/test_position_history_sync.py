from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from debco.live.position_history_sync import enrich_closed_positions_from_history
from debco.live.state_store import LiveStateStore


class FakeMT5:
    DEAL_ENTRY_IN = 0
    DEAL_ENTRY_OUT = 1


class FakeClient:
    mt5 = FakeMT5()

    def history_deals_get(self, date_from, date_to, position=None):
        assert position == 111
        return [
            SimpleNamespace(ticket=1, position_id=111, entry=FakeMT5.DEAL_ENTRY_IN, time=1000, price=1.1, profit=0.0),
            SimpleNamespace(ticket=2, position_id=111, entry=FakeMT5.DEAL_ENTRY_OUT, time=1100, price=1.2, profit=-3.5),
        ]


def test_enrich_closed_position_from_history(tmp_path: Path):
    state = LiveStateStore(tmp_path / "state.sqlite")
    state.upsert_position({
        "mt5_position_ticket": "111",
        "symbol": "EURUSD",
        "setup_id": "EUR_AH_ATR2_BUY",
        "side": "long",
        "magic": 130103,
        "volume": 0.1,
        "status": "open",
    })
    state.mark_position_closed(mt5_position_ticket="111", close_reason="not_found_in_mt5_positions", status="externally_closed_or_tp_sl")
    result = enrich_closed_positions_from_history(state, FakeClient())
    assert result.enriched_positions == 1
    with state.connect() as con:
        row = dict(con.execute("SELECT * FROM positions WHERE mt5_position_ticket='111'").fetchone())
    assert row["profit"] == -3.5
    assert row["close_price"] == 1.2
