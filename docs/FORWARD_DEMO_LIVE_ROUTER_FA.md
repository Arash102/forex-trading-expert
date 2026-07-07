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

## v0.1.13b — اتصال inference زنده

در این مرحله router همچنان به صورت dry-run اجرا می‌شود و سفارش واقعی ارسال نمی‌کند، اما مسیر inference واقعی اضافه شده است:

- ساخت feature زنده از کندل بسته‌شده M15؛
- ساخت/merge کردن DXY معکوس برای featureهای market-relative؛
- اعمال candidate filter همان setup inventory؛
- بارگذاری مدل نهایی هر setup از `data/live_models/<setup_id>/`؛
- تبدیل policyهای `top_percentile_by_fold` به cutoff ثابت قابل اجرای live؛
- تولید `order_intent` فقط وقتی probability از cutoff عبور کند.

### ترتیب اجرا

ابتدا مدل‌های live را بسازید:

```bash
python scripts/13_train_live_models.py --ml-config configs/ml_config.local.json --live-spec data/final_strategy_report/live_execution_spec.json --models-dir data/live_models
```

بعد اعتبارسنجی کنید:

```bash
python scripts/13_validate_live_models.py --live-spec data/final_strategy_report/live_execution_spec.json --models-dir data/live_models
```

سپس router را در حالت inference dry-run اجرا کنید:

```bash
python scripts/12_forward_demo_router.py --live-config configs/live_router.example.json --once --enable-inference
```

در config پیش‌فرض، `inference.enabled=false` است تا router بدون مدل هم ایمن اجرا شود. برای smoke test می‌توان از `--enable-inference` استفاده کرد. برای اجرای مداوم بعدی، مقدار `inference.enabled` را در یک config محلی به `true` تغییر دهید.

### نکته مهم درباره top-percentile

در live نمی‌توان از top-percentile آینده استفاده کرد. بنابراین `13_train_live_models.py` برای setupهایی که policy آنها `top_percentile_by_fold` است، cutoff ثابت را از توزیع OOF همان setup محاسبه و در `artifact.json` ذخیره می‌کند. از این به بعد live router فقط با `live_probability_cutoff` کار می‌کند.

### محدودیت این مرحله

این مرحله هنوز order واقعی نمی‌زند. خروجی فقط signal/order-intent dry-run و chart event است. فعال‌سازی order واقعی باید در milestone بعدی و بعد از بررسی smoke test انجام شود.

## v0.1.13c — اجرای سفارش دمو با محافظ‌های سخت

در این مرحله مسیر ارسال سفارش به MT5 اضافه می‌شود، اما همچنان پیش‌فرض سیستم `dry_run=true` است. سفارش واقعی فقط وقتی ارسال می‌شود که کاربر در همان اجرا فلگ صریح زیر را بدهد:

```bash
python scripts/12_forward_demo_router.py --live-config configs/live_router.example.json --once --enable-inference --enable-demo-orders
```

محافظ‌های اصلی:

- `execution.enable_orders` باید true شود؛
- `execution.dry_run` باید false شود؛
- فلگ runtime یعنی `--enable-demo-orders` باید داده شود؛
- `execution.demo_only=true` باید باقی بماند؛
- حساب MT5 باید از نوع demo تشخیص داده شود؛
- در غیر این صورت order به جای ارسال، با وضعیت blocked در SQLite ثبت می‌شود.

### محاسبه حجم معامله

برای هر signal، سیستم از `job` همان setup مقدارهای TP/SL/horizon را استخراج می‌کند. نمونه:

```text
EURUSD_fast_15_8_h16_long  -> TP=15 pip, SL=8 pip, horizon=16 کندل
XAUUSD_runner_2200_1100_h40_long -> TP=2200 pip, SL=1100 pip, horizon=40 کندل
```

حجم معامله با اطلاعات خود MT5 محاسبه می‌شود:

```text
risk_amount = account_equity × risk_per_trade × risk_weight
risk_per_lot = stop_distance / trade_tick_size × trade_tick_value
volume = risk_amount / risk_per_lot
```

سپس volume بر اساس `volume_min`، `volume_max` و `volume_step` نماد در MT5 normalize می‌شود.

### وزن ریسک XAU sell

risk plan منتخب `xau_sell_50pct` از `live_execution_spec.json` خوانده می‌شود. بنابراین برای setupهای فروش XAUUSD، ریسک مؤثر نصف می‌شود:

```text
XAUUSD|short -> risk_weight = 0.5
```

مثلاً اگر ریسک پایه ۱٪ باشد، فروش‌های XAU با ۰.۵٪ equity محاسبه می‌شوند.

### TP/SL و magic number

در سفارش MT5:

- نوع سفارش market است؛
- برای long از ask و برای short از bid استفاده می‌شود؛
- TP/SL همان لحظه order placement داخل request قرار می‌گیرد؛
- magic number همان magic اختصاصی setup است؛
- comment با prefix `DEBCO` و نام setup ثبت می‌شود.

### اجرای امن smoke test

ابتدا همیشه dry-run را تست کن:

```bash
python scripts/12_forward_demo_router.py --live-config configs/live_router.example.json --once --enable-inference --inject-test-signal EUR_AH_ATR2_BUY
```

بعد اگر حساب MT5 واقعاً demo بود، فقط برای تست دمو:

```bash
python scripts/12_forward_demo_router.py --live-config configs/live_router.example.json --once --enable-inference --enable-demo-orders --inject-test-signal EUR_AH_ATR2_BUY
```

در حساب real، همین دستور باید block شود و order ارسال نکند.
