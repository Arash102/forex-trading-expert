# قدم‌های ساده برای رساندن پروژه به تست دمو یک‌ماهه

این نسخه برای ساده‌کردن اجراست. دیگر لازم نیست دستی چند config را یکی‌یکی اصلاح کنی.

## 1) آماده‌سازی configهای local

```bash
python scripts/16b_prepare_demo_month_local_configs.py --login 878713 --risk 0.01
```

این دستور:

- از `configs/live_router.local.json` بکاپ می‌گیرد.
- همان فایل را برای demo-month نهایی می‌کند.
- فایل `configs/demo_month_launch_lock.local.json` را با تاییدهای لازم می‌سازد.
- رمز، server و terminal_path موجود در `live_router.local.json` را دست‌کاری نمی‌کند.

## 2) تشخیص ساده مشکل فیچرها

```bash
python scripts/17_diagnose_live_features.py --live-config configs/live_router.local.json
```

اگر خروجی گفت `DXY_TIME_LAG` یا `DXY_TIME_ALIGNMENT`، یعنی مشکل از مدل نیست؛ مشکل از این است که DXY برای کندل بسته‌شده نماد هنوز exact match نشده است.

## 3) healthcheck اصلی

```bash
python scripts/12_forward_demo_router.py --live-config configs/live_router.local.json --startup-healthcheck-only --enable-inference --enable-demo-orders
```

فقط اگر خروجی این شد ادامه بده:

```text
STARTUP READY
```

## 4) readiness و launch lock

تا وقتی روی branch هستی:

```bash
python scripts/15_validate_demo_readiness.py --live-config configs/live_router.local.json
python scripts/16_validate_demo_month_launch_lock.py --live-config configs/live_router.local.json --launch-lock configs/demo_month_launch_lock.local.json --skip-git-check
```

بعد از merge روی `main`، دستور launch lock را بدون `--skip-git-check` بگیر.

## 5) اجرای دمو

```bash
python scripts/12_forward_demo_router.py --live-config configs/live_router.local.json --enable-inference --enable-demo-orders
```

## قانون مهم

تا وقتی مرحله 3 خروجی `STARTUP READY` ندهد، اجرای یک‌ماهه را شروع نکن.
