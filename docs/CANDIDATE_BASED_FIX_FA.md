# اصلاح candidate-based v0.1.6

این patch دو اصلاح عملی اضافه می‌کند:

1. امکان اجرای train روی یک job مشخص:

```bash
python scripts/05_train_candidate_sets.py --config configs/ml_config.local.json --candidate-set session_volatility_v1 --job EURUSD_fast_15_8_h16_short
```

2. جلوگیری از calibration وارونه در sigmoid:

اگر calibration tail پرنویز باشد، LogisticRegression ممکن است ضریب منفی یاد بگیرد و احتمال خام بالاتر را به احتمال calibrated پایین‌تر تبدیل کند. در این حالت calibration به صورت پیش‌فرض به raw fallback می‌کند و reason برابر `sigmoid_inverted_fallback` می‌شود.

در config:

```json
"calibration": {
  "allow_inverted_sigmoid": false
}
```

برای سیگنال معاملاتی، `y_prob_raw` همچنان معیار اصلی ranking است. `y_prob_calibrated` فقط وقتی قابل استفاده است که وارونه نشده باشد و threshold مناسب خودش را داشته باشد.


## Metric threshold correction

برای candidate-based، metric های fold-level از این پس روی `y_prob_raw` و threshold اولیه `0.30` گزارش می‌شوند. دلیل این تغییر این است که calibrated probabilities اغلب حول base-rate فشرده می‌شوند و threshold ثابت `0.60` می‌تواند fold_precision را مصنوعا صفر کند. معیار انتخاب نهایی همچنان باید `threshold_sweep.csv` و سپس trading evaluation باشد.
