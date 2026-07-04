# Candidate-based metric threshold fix

در خروجی‌های candidate-based نسخه قبلی، ستون‌های `precision` و `recall` داخل `fold_metrics.csv` برای بسیاری از fold ها صفر می‌شدند. علت اصلی این نبود که مدل هیچ سیگنالی ندارد؛ علت این بود که `fold_metrics` با threshold ثابت `0.60` و معمولا روی `y_prob_calibrated` محاسبه می‌شد.

در candidate-based، sigmoid calibration احتمال‌ها را به سمت base-rate فشرده می‌کند. بنابراین threshold 0.60 روی احتمال calibrated معمولا بسیار سختگیرانه است و می‌تواند تقریبا همه سیگنال‌ها را صفر کند، در حالی که `threshold_sweep.csv` نشان می‌دهد threshold های پایین‌تر مثل 0.25 تا 0.40 سیگنال و MCC مثبت دارند.

اصلاح انجام شده:

```json
"model": {
  "threshold": 0.3,
  "evaluation_probability_column": "y_prob_raw"
}
```

بنابراین از این نسخه به بعد:

- `fold_metrics.csv` و لاگ حین train با `y_prob_raw` و threshold اولیه 0.30 محاسبه می‌شوند.
- `y_prob_calibrated` همچنان در `oof_predictions.csv` ذخیره می‌شود.
- `threshold_sweep.csv` همچنان هم `y_prob_raw` و هم `y_prob_calibrated` را مقایسه می‌کند.
- calibration برای Brier/ECE و تحلیل احتمال حفظ می‌شود، اما دیگر به طور پیش فرض باعث صفر شدن سیگنال‌های fold metrics نمی‌شود.
