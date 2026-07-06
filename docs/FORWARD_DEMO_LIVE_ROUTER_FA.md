# v0.1.13a — Forward/Demo Live Router

این مرحله نسخه اجرایی اولیه را می‌سازد، اما هنوز برای سفارش واقعی فعال نیست. هدف این milestone:

- خواندن `live_execution_spec.json`
- اعتبارسنجی ۱۲ setup منتخب و magic number هر setup
- تشخیص کندل جدید M15 با Python
- پردازش فقط روی کندل بسته‌شده قبلی
- ثبت state در SQLite
- تولید event برای marker و screenshot در MT5
- dry-run / signal-only به عنوان مرحله امن اول

## اصل معماری

Python مغز تصمیم‌گیری است. MT5 فقط دیتای بازار، اجرای سفارش، و نگهداری پوزیشن را انجام می‌دهد. MQL helper هیچ rule، feature، مدل یا سیگنال معاملاتی حساب نمی‌کند؛ فقط marker و screenshot می‌سازد.

## زمان‌بندی

Router از روش event-aware polling استفاده می‌کند:

- در زمان عادی هر چند ثانیه MT5 را چک می‌کند.
- نزدیک زمان کندل جدید، polling سریع‌تر می‌شود.
- به محض اینکه کندل جدید در MT5 دیده شد، کندل قبلی را بسته‌شده فرض می‌کند و فقط همان را پردازش می‌کند.
- برای هر `symbol + setup_id + side + signal_bar_time` فقط یک تصمیم ثبت می‌شود.

## Magic number

هر setup یک magic number مستقل دارد. این mapping در `configs/live_router.example.json` ذخیره شده و نباید در کد hardcode شود.

## MQL helper

فایل `mql5/DEBCO_ChartEventHelper.mq5` را در MT5 داخل مسیر Experts قرار بده و روی چارت EURUSD و XAUUSD اجرا کن. این helper فایل‌های `.cmd` را می‌خواند، دایره سبز entry و نام setup را روی چارت می‌گذارد و screenshot می‌گیرد.

## اجرای اعتبارسنجی

```bash
python scripts/12_validate_live_router_spec.py --live-config configs/live_router.example.json
```

## اجرای dry-run

```bash
python scripts/12_forward_demo_router.py --live-config configs/live_router.example.json
```

برای smoke test marker/screenshot بدون سیگنال واقعی:

```bash
python scripts/12_forward_demo_router.py --live-config configs/live_router.example.json --once --inject-test-signal EUR_AH_ATR2_BUY
```

## محدودیت فعلی

در v0.1.13a، inference واقعی مدل‌ها عمدا فعال نشده است. این نسخه زیرساخت timing/state/magic/chart-event را تثبیت می‌کند. مرحله بعد اتصال feature/model inference و سپس demo order execution است.
