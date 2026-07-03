# XGBoost + Optuna + Calibration + Threshold Sweep

این مرحله روی فایل‌های strict ML-ready کار می‌کند:

```text
data/ml_ready/*_ml_ready_*.csv
```

این فایل‌ها فقط شامل 100 فیچر نهایی و ستون `label` هستند. فایل‌های metadata فقط برای split زمانی، audit و گزارش استفاده می‌شوند و وارد مدل نمی‌شوند.

## خروجی‌های جدید v0.1.5

برای هر job در مسیر زیر خروجی ساخته می‌شود:

```text
data/ml_results/<experiment>/<symbol>_<profile>_<side>/
```

فایل‌های مهم:

```text
oof_predictions.csv
fold_metrics.csv
metrics_summary.csv
threshold_sweep.csv
calibration_summary.csv
best_params_by_fold.csv
run_config.json
```

`oof_predictions.csv` خروجی out-of-fold است و برای تحلیل آستانه، calibration و بعدا بک تست معاملاتی استفاده می‌شود. ستون‌های مهم:

```text
fold
row_idx
y_true
y_prob_raw
y_prob_calibrated
y_pred
date / entry_date / exit_date از metadata
```

## Calibration

در config پیش فرض، calibration فعال است:

```json
"calibration": {
  "enabled": true,
  "method": "sigmoid"
}
```

برای هر fold، train به دو بخش تقسیم می‌شود:

```text
model_train
calibration_tail
```

مدل روی `model_train` آموزش می‌بیند و calibrator روی `calibration_tail` fit می‌شود. test fold دست نخورده می‌ماند.

## Threshold Sweep

برای هر job و هر fold، آستانه‌های مختلف روی احتمال خام و احتمال calibrated تست می‌شود:

```text
0.40 تا 0.85
```

متریک‌ها شامل precision، recall، specificity، balanced accuracy، MCC و signal rate هستند.

## Purged Walk-forward

در walk-forward، `purge_bars` اضافه شده است. این مقدار ردیف‌های آخر train قبل از شروع test را حذف می‌کند تا overlap ناشی از horizon لیبل Triple Barrier کاهش یابد.

```json
"purge_bars": 40
```

برای شروع از 40 استفاده شده، چون بیشترین horizon فعلی XAUUSD برابر 40 کندل است.

## Candidate-based / Meta-labeling

در config یک hook برای candidate-based training اضافه شده است، اما پیش فرض خاموش است:

```json
"candidate_filter": {
  "enabled": false
}
```

وقتی روشن شود، مدل فقط روی کندل‌هایی آموزش می‌بیند که primary candidate filter آنها را مناسب دانسته است. این مسیر برای مرحله meta-labeling است: اول candidate بساز، بعد XGBoost فقط تایید کند کدام candidate ارزش ورود دارد.

فعلا baseline روی همه کندل‌ها اجرا می‌شود. اگر edge ضعیف بود، در مرحله بعد candidate filter را فعال و تنظیم می‌کنیم.
