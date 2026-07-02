# Feature Engineering Core v0.1.2 — Symbol-Aware

این مرحله feature engineering را از حالت global خارج می‌کند و برای هر نماد config جدا می‌سازد.

## اصل‌ها

- Python تنها منبع حقیقت برای فیچرهاست.
- `pip_size`, `point_size`, `zigzag_deviation_pct`, threshold های bias و threshold روند H1 برای هر نماد جدا هستند.
- خروجی کامل و خروجی مدل جدا هستند:
  - `data/features/full/<SYMBOL>_features_full.csv`
  - `data/features/model/<SYMBOL>_features_model.csv`
- خروجی model فقط ستون‌های انتخاب شده و lag های مجاز را دارد و سقف آن در config کنترل می‌شود.
- هیچ خروجی داخل `data/` وارد Git نمی‌شود.

## موارد symbol-aware فعلی

| نماد | pip size | point size | zigzag deviation pct | prev/day bias threshold |
|---|---:|---:|---:|---:|
| EURUSD | 0.0001 | 0.00001 | 0.13 | 8 pip |
| XAUUSD | 0.01 | 0.01 | 0.8 | 250 pip |

## خروجی‌ها

`full` برای تحقیق، rule-base، debug و بررسی دستی است. این فایل می‌تواند ستون‌های زیادی داشته باشد.

`model` برای XGBoost است. در این نسخه حداکثر تعداد فیچرهای مدل، شامل lag ها، 100 تنظیم شده است.

## نکته درباره session features

فیچرهای کامل‌شده هر سشن فقط بعد از پایان همان سشن پر می‌شوند تا leakage نداشته باشیم. برای داخل همان سشن، فیچرهای `current_session_*_so_far` ساخته می‌شوند.

## اجرای مرحله

```bash
python scripts/02_build_features.py --data-config configs/data_config.local.json --features-config configs/features_config.local.json
python scripts/02b_sanity_check_features.py --features-config configs/features_config.local.json
```

## نکته درباره نام ستون‌ها

نام ستون‌ها در کد و CSV همگی lowercase هستند، مثلا:

- `is_monday`
- `is_friday`

نه `is_Monday` یا `is_Friday`.
