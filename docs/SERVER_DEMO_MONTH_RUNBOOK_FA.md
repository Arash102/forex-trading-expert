# راهنمای اجرای یک ماهه دمو روی سرور

این milestone فقط لایه عملیاتی را قوی می کند و به مدل، setup، threshold، SL/TP یا منطق معاملاتی دست نمی زند.

## قبل از اجرا روی سرور

روی سرور از tag قفل شده استفاده کن:

```bash
git clone https://github.com/Arash102/forex-trading-expert.git
cd forex-trading-expert
git checkout v0.1.14a-demo-month-launch-ready
```

بعد فایل های local را منتقل کن:

```text
configs/live_router.local.json
configs/demo_month_launch_lock.local.json
configs/live_ops.local.json
```

فایل `configs/live_ops.local.json` را از روی `configs/live_ops.example.json` بساز.

## پوشه هایی که نباید پاک شوند

```text
data/live_state
data/live_reports
data/live_diagnostics
data/live_runtime
```

اگر برنامه یا سرور restart شود، state قبلی از `data/live_state` خوانده می شود و نباید معاملات قبلی گم شوند.

## اجرای پیشنهادی روی Windows Server

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_demo_month_server.ps1
```

این wrapper قبل از شروع router سه gate را اجرا می کند:

```text
17_diagnose_live_features.py
12_forward_demo_router.py --startup-healthcheck-only
16_validate_demo_month_launch_lock.py
```

بعد router اصلی را با `--enable-inference --enable-demo-orders` اجرا می کند.

## روز تعطیل و بازار بسته

در بازار بسته، برنامه نباید معامله کند. رفتار طبیعی این است:

```text
no_new_bar / market idle
```

اگر wrapper روشن باشد، heartbeat همچنان آپدیت می شود و watchdog نباید هشدار crash بدهد.

## اگر گزارش position اشتباه ماند

برای sync گزارش بدون ارسال order جدید:

```bash
python scripts/18_sync_live_positions_report_only.py --live-config configs/live_router.local.json
```

این دستور فقط positionها را از MT5 و history می خواند و گزارش روزانه را دوباره می نویسد. هیچ order جدیدی ارسال نمی کند.

## Watchdog

برای چک کردن heartbeat:

```bash
python scripts/19_watchdog_live_router.py --ops-config configs/live_ops.local.json
```

برای اجرای پیوسته:

```bash
python scripts/19_watchdog_live_router.py --ops-config configs/live_ops.local.json --loop --poll-seconds 60
```

## Alert

همیشه alertها در فایل زیر نوشته می شوند:

```text
data/live_runtime/alerts.jsonl
```

برای Telegram، در `configs/live_ops.local.json` مقدار `telegram.enabled` را true کن و این env var ها را روی سرور ست کن:

```text
DEBCO_TELEGRAM_BOT_TOKEN
DEBCO_TELEGRAM_CHAT_ID
```

## قانون فاز دمو

تا پایان یک ماه دمو:

```text
مدل تغییر نمی کند
setup تغییر نمی کند
threshold تغییر نمی کند
risk logic تغییر نمی کند
فقط reporting / sync / heartbeat / watchdog / alert اصلاح می شود
```
