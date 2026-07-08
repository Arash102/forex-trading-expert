# Runbook اجرای یک‌ماهه حساب دمو DEBCO

این سند برای اجرای کنترل‌شده نسخه دمو است. در طول دوره دمو، تغییر استراتژی، فیچر، مدل، threshold و setup ممنوع است. فقط باگ‌فیکس، گزارش‌گیری، guard و مستندسازی مجاز است.

## 1) ساخت branch

```bash
git checkout main
git pull origin main
git checkout -b feat/demo-month-launch-lock
```

## 2) ساخت فایل lock محلی

```bash
cp configs/demo_month_launch_lock.example.json configs/demo_month_launch_lock.local.json
```

در فایل local این موارد را فقط بعد از بررسی دستی true کن:

```json
"acknowledgements": {
  "strategy_frozen": true,
  "features_frozen": true,
  "models_frozen": true,
  "demo_account_verified": true,
  "manual_preflight_done": true,
  "risk_acceptance": true
}
```

اگر اسم سرور دمو بروکر شامل `demo` نیست، مقدار `account.server_must_contain_any` را با کلیدواژه درست سرور دمو یا `account.allowed_logins` تنظیم کن.

## 3) تنظیم live_router.local.json برای اجرای واقعی روی دمو

برای اجرای یک‌ماهه دمو، بخش‌های کلیدی باید این‌طور باشند:

```json
"execution": {
  "dry_run": false,
  "enable_orders": true,
  "demo_only": true,
  "require_demo_orders_cli_flag": true,
  "risk_per_trade": 0.01,
  "horizon_exit_enabled": true,
  "one_signal_per_setup_per_bar": true
},
"inference": {
  "enabled": true,
  "require_live_models_for_all_setups": true
},
"position_manager": {
  "enabled": true,
  "sync_open_positions": true,
  "horizon_exit_enabled": true
},
"reports": {
  "enabled": true,
  "write_on_each_new_bar": true
}
```

## 4) تست‌ها

```bash
python -m pytest tests/test_forward_demo_router.py tests/test_forward_demo_live_inference.py tests/test_forward_demo_order_execution.py tests/test_forward_demo_position_manager.py tests/test_forward_demo_guards_reporting.py tests/test_demo_month_launch_lock.py
```

## 5) سه gate قبل از شروع دمو

```bash
python scripts/12_forward_demo_router.py --live-config configs/live_router.local.json --startup-healthcheck-only --enable-inference --enable-demo-orders
```

```bash
python scripts/15_validate_demo_readiness.py --live-config configs/live_router.local.json
```

```bash
python scripts/16_validate_demo_month_launch_lock.py --live-config configs/live_router.local.json --launch-lock configs/demo_month_launch_lock.local.json
```

اگر هر سه OK شدند، اجرای دمو مجاز است.

## 6) شروع اجرای دمو

ابتدا یک اجرای کوتاه و supervised انجام بده:

```bash
python scripts/12_forward_demo_router.py --live-config configs/live_router.local.json --enable-inference --enable-demo-orders
```

بعد از اطمینان از لاگ، مارکر، گزارش و عدم خطا، همین دستور را در محیط پایدار اجرا کن.

## 7) قوانین توقف اضطراری

اجرای دمو باید متوقف شود اگر یکی از موارد زیر رخ داد:

- اتصال MT5 ناپایدار شد یا خطاهای پیاپی از سقف config گذشت.
- position manager با پوزیشن‌های واقعی حساب sync نشد.
- گزارش‌ها یا state db نوشته نشدند.
- تعداد معاملات روزانه، معاملات باز، یا معاملات هم‌جهت/خلاف‌جهت از guard عبور کرد.
- حساب اشتباهی، سرور اشتباهی یا لاگین اشتباهی تشخیص داده شد.
- رفتار order execution با انتظار متفاوت بود.

## 8) Git بعد از پاس شدن milestone

```bash
git add configs/demo_month_launch_lock.example.json docs/DEMO_MONTH_RUNBOOK_FA.md scripts/16_validate_demo_month_launch_lock.py src/debco/live/demo_launch_lock.py tests/test_demo_month_launch_lock.py
git commit -m "feat: add demo month launch lock validation"
git checkout main
git merge feat/demo-month-launch-lock
git tag v0.1.14a-demo-month-launch-lock-validation-ok
git push origin main --tags
```
