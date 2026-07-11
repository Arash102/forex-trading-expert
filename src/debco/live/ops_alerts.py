from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(payload), ensure_ascii=False, allow_nan=False) + "\n")


@dataclass(frozen=True)
class AlertConfig:
    jsonl_path: str = "data/live_runtime/alerts.jsonl"
    telegram_enabled: bool = False
    telegram_bot_token_env: str = "DEBCO_TELEGRAM_BOT_TOKEN"
    telegram_chat_id_env: str = "DEBCO_TELEGRAM_CHAT_ID"
    throttle_seconds: int = 1800

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "AlertConfig":
        data = dict(cfg or {})
        telegram = dict(data.get("telegram") or {})
        return cls(
            jsonl_path=str(data.get("jsonl_path", data.get("alert_jsonl_path", "data/live_runtime/alerts.jsonl"))),
            telegram_enabled=bool(telegram.get("enabled", data.get("telegram_enabled", False))),
            telegram_bot_token_env=str(telegram.get("bot_token_env", "DEBCO_TELEGRAM_BOT_TOKEN")),
            telegram_chat_id_env=str(telegram.get("chat_id_env", "DEBCO_TELEGRAM_CHAT_ID")),
            throttle_seconds=int(data.get("throttle_seconds", 1800)),
        )


class AlertManager:
    """Small operational alert sink.

    It always writes JSONL locally. Telegram is optional and controlled by env vars;
    no trading logic depends on alert delivery.
    """

    def __init__(self, config: AlertConfig | Mapping[str, Any] | None = None):
        self.config = config if isinstance(config, AlertConfig) else AlertConfig.from_mapping(config)
        self._last_sent: dict[str, float] = {}

    def send(
        self,
        *,
        level: str,
        event: str,
        message: str,
        details: Mapping[str, Any] | None = None,
        throttle_key: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        now = time.time()
        key = throttle_key or f"{level}:{event}:{message}"
        throttled = False
        last = self._last_sent.get(key)
        if not force and last is not None and (now - last) < self.config.throttle_seconds:
            throttled = True
        else:
            self._last_sent[key] = now

        payload = {
            "alert_id": uuid4().hex,
            "created_at_utc": utc_now_iso(),
            "level": str(level).upper(),
            "event": str(event),
            "message": str(message),
            "throttle_key": key,
            "throttled": throttled,
            "details": dict(details or {}),
        }
        _append_jsonl(self.config.jsonl_path, payload)

        if not throttled and self.config.telegram_enabled:
            try:
                payload["telegram_sent"] = self._send_telegram(payload)
            except Exception as exc:  # alert failure must never break trading ops
                payload["telegram_error"] = repr(exc)
                _append_jsonl(self.config.jsonl_path, {**payload, "event": "telegram_send_error"})
        return payload

    def _send_telegram(self, payload: Mapping[str, Any]) -> bool:
        token = os.environ.get(self.config.telegram_bot_token_env, "").strip()
        chat_id = os.environ.get(self.config.telegram_chat_id_env, "").strip()
        if not token or not chat_id:
            return False
        text = (
            f"DEBCO {payload.get('level')} | {payload.get('event')}\n"
            f"{payload.get('message')}\n"
            f"time={payload.get('created_at_utc')}"
        )
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec - user-provided token endpoint
            return 200 <= int(resp.status) < 300
