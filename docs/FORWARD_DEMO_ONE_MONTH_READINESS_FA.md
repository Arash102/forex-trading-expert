# چک‌لیست اجرای یک‌ماهه دمو DEBCO

## قبل از شروع

1. MT5 روی حساب demo باز باشد.
2. نمادهای EURUSD و XAUUSD در Market Watch فعال باشند.
3. مدل‌های live در `data/live_models` valid باشند.
4. `configs/live_router.local.json` ساخته شده باشد.
5. MQL helper روی چارت‌های EURUSD و XAUUSD نصب شده باشد.
6. حداقل یک اجرای supervised بدون `--inject-test-signal` انجام شود.

## دستور آماده‌سازی config local

```bash
python scripts/14_create_live_router_local_config.py --risk-per-trade 0.01
python scripts/15_validate_demo_readiness.py --live-config configs/live_router.local.json
```

## اجرای یک‌ماهه دمو

```bash
python scripts/12_forward_demo_router.py --live-config configs/live_router.local.json --enable-inference --enable-demo-orders
```

## توقف اضطراری

1. ترمینال را با `Ctrl+C` متوقف کن.
2. اگر لازم بود positionها را از داخل MT5 دستی ببند.
3. فایل‌های `data/live_reports` و `data/live_state/forward_demo.sqlite` را برای audit نگه دار.

## فایل‌هایی که نباید commit شوند

- `configs/live_router.local.json`
- `data/live_state/`
- `data/live_reports/`
- `data/live_models/`

## بازآموزی مدل بعد از ۱ تا ۳ ماه

بازآموزی نباید داخل router live انجام شود. مسیر درست:

1. دیتای جدید را از MT5 بگیر.
2. pipeline research/training را offline اجرا کن.
3. challenger model را با champion مقایسه کن.
4. فقط در صورت قبولی PF/RoR/DD، مدل جدید را promote کن.
