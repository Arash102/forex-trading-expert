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

# v0.1.6 Candidate-based / Meta-labeling

در نسخه `v0.1.6` مدل دیگر مجبور نیست فقط روی همه کندل‌ها آموزش ببیند. یک لایه candidate filter اضافه شده است تا اول کندل‌های قابل معامله انتخاب شوند و بعد XGBoost نقش meta model یا تایید کننده کیفیت setup را داشته باشد.

اصل مهم این مرحله این است:

```text
candidate filter فقط از فیچرهای همان کندل استفاده می‌کند
label / outcome / realized_pips / entry و exit وارد candidate filter نمی‌شوند
```

## اجرای sanity قبل از train

قبل از آموزش گران XGBoost، باید ببینیم هر candidate set چند ردیف نگه می‌دارد و positive rate آن بهتر شده یا نه:

```bash
python scripts/05a_sanity_check_candidates.py --config configs/ml_config.local.json --save
```

خروجی مهم:

```text
rows_after
keep_ratio
positive_rate_before
positive_rate_after
positive_lift
fold_count_after
```

اگر یک candidate set کمتر از حدود 1 تا 3 درصد داده را نگه دارد یا fold کافی نسازد، برای آموزش پایدار مناسب نیست.

## Candidate sets فعلی

در config چند candidate set تعریف شده است:

```text
session_volatility_v1
```
فیلتر عمومی بر اساس سشن فعال، اسپرد، ATR percentile، volatility و range جاری سشن.

```text
directional_trend_v1
```
فیلتر روندی با GMMA، جهت H1 و RSI. برای long و short شرط‌ها side-aware هستند.

```text
session_breakout_v1
```
فیلتر شکست ساختار آسیا/لندن. برای long از شکست Asia high و برای short از شکست Asia low استفاده می‌کند.

```text
trend_pullback_v1
```
فیلتر pullback در جهت روند. ایده آن این است که قیمت در کانتکست روندی باشد اما خیلی از ساختار swing فاصله نگرفته باشد.

```text
mean_reversion_v1
```
فیلتر برگشتی نزدیک swing high/low با RSI نسبتا افراطی.

## آموزش candidate sets

برای آموزش همه candidate set های فعال:

```bash
python scripts/05_train_candidate_sets.py --config configs/ml_config.local.json
```

برای شروع سریع‌تر، بهتر است یک candidate set را جدا اجرا کنیم:

```bash
python scripts/05_train_candidate_sets.py --config configs/ml_config.local.json --candidate-set session_volatility_v1
```

یا برای smoke test فقط دو job اول:

```bash
python scripts/05_train_candidate_sets.py --config configs/ml_config.local.json --candidate-set session_volatility_v1 --max-jobs 2
```

## مقایسه candidate sets

بعد از train:

```bash
python scripts/05b_compare_candidate_sets.py --config configs/ml_config.local.json
```

خروجی تجمیعی:

```text
data/ml_results/candidate_global_v0_1_6/candidate_set_comparison.csv
```

## معیار موفقیت candidate-based

Candidate-based فقط وقتی بهتر است که نسبت به baseline all-candles حداقل بخشی از اینها بهتر شود:

```text
positive_rate_after > positive_rate_before
positive_lift > 1
MCC بالاتر
precision lift بهتر
signal_rate قابل قبول
fold_count کافی
```

اگر candidate set از نظر ML بهتر شد، مرحله بعدی ارزیابی معاملاتی OOF است؛ یعنی از `oof_predictions.csv` و metadata برای محاسبه win rate، profit factor، net pips و drawdown استفاده می‌کنیم.


## اصلاح اعتبارسنجی در candidate-based training

در نسخه candidate-based نباید ابتدا دیتاست را به candidate rows فشرده کنیم و بعد walk-forward بسازیم. این کار فاصله زمانی واقعی را از بین می برد و ممکن است تعداد fold ها را از ۹ به ۰ یا ۱ کاهش دهد.

روش درست این است:

```text
original chronological timeline -> walk-forward folds
inside each fold -> keep only candidate rows for train/test
```

بنابراین `candidate_validation.mode = base_timeline` پیش فرض است. مدل فقط روی candidate rows آموزش می بیند و فقط روی candidate rows تست می شود، اما مرزهای زمانی fold ها از تایم لاین اصلی ساخته می شوند.
