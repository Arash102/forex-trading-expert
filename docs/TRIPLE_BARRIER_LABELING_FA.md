# برچسب گذاری Triple Barrier نسخه v0.1.3

این مرحله بعد از ساخت فیچرهای model اجرا می‌شود و برای هر نماد، هر پروفایل معاملاتی و هر جهت، یک target جدا می‌سازد.

اصل طراحی این نسخه:

- `long_target`: آیا ورود long با TP/SL/horizon مشخص، به TP می‌رسد؟
- `short_target`: آیا ورود short با TP/SL/horizon مشخص، به TP می‌رسد؟
- `label = 1`: TP جهت انتخاب شده اول لمس شده است.
- `label = 0`: SL، vertical barrier، no-entry یا حالت neutral رخ داده است.
- `outcome_label` برای audit نگه داشته می‌شود: `+1` برای TP، `-1` برای SL و `0` برای vertical یا neutral.

## پروفایل‌های شروع

برای EURUSD دو پروفایل ساخته می‌شود:

```text
runner_20_10_h24: TP=20 pip, SL=10 pip, horizon=24 M15 bars
fast_15_8_h16:   TP=15 pip, SL=8 pip,  horizon=16 M15 bars
```

برای XAUUSD دو پروفایل ساخته می‌شود:

```text
runner_2200_1100_h40: TP=2200 pip, SL=1100 pip, horizon=40 M15 bars
active_1500_800_h32:  TP=1500 pip, SL=800 pip,  horizon=32 M15 bars
```

با `pip_size = 0.01` برای XAUUSD، مقدار `2200 pip` یعنی حدود 22 دلار حرکت قیمت.

## زمان ورود

پیش فرض فعلی:

```json
"entry_offset_bars": 1,
"entry_price_column": "open"
```

یعنی فیچرهای کندل `t` بعد از بسته شدن کندل شناخته می‌شوند و ورود فرضی روی open کندل بعدی انجام می‌شود. این برای استفاده live محافظه کارانه‌تر و causal است.

## سیاست کندل مبهم

اگر در یک کندل هم TP و هم SL قابل لمس باشد، با OHLC نمی‌دانیم کدام اول رخ داده است. پیش فرض:

```json
"same_bar_policy": "sl_first"
```

این سیاست محافظه کارانه است و جلوی خوش بینی کاذب در labeling را می‌گیرد.

## فایل‌های خروجی

برای هر نماد، هر پروفایل و هر جهت این فایل‌ها ساخته می‌شود:

```text
data/labels/EURUSD_runner_20_10_h24_labels_long.csv
data/labels/EURUSD_runner_20_10_h24_labels_short.csv
data/labels/EURUSD_fast_15_8_h16_labels_long.csv
data/labels/EURUSD_fast_15_8_h16_labels_short.csv

data/labels/XAUUSD_runner_2200_1100_h40_labels_long.csv
data/labels/XAUUSD_runner_2200_1100_h40_labels_short.csv
data/labels/XAUUSD_active_1500_800_h32_labels_long.csv
data/labels/XAUUSD_active_1500_800_h32_labels_short.csv
```

اگر `write_joined_dataset = true` باشد، نسخه join شده با فیچرهای model هم ساخته می‌شود و ورودی مرحله XGBoost خواهد بود.

## منطق استفاده در decision layer

در مرحله ML دو مدل یا دو target داریم:

```text
p_long  = احتمال long_target=1
p_short = احتمال short_target=1
```

قاعده عملیاتی بعدی می‌تواند این باشد:

```text
اگر p_long بالا و p_short پایین باشد: long candidate
اگر p_short بالا و p_long پایین باشد: short candidate
اگر هر دو پایین باشند: no trade
اگر هر دو بالا یا نزدیک هم باشند: conflict / no trade
```

این decision layer هنوز در این milestone پیاده سازی نمی‌شود؛ این مرحله فقط label و dataset آموزشی را می‌سازد.

## اصل مهم پروژه

هیچ label، feature یا تصمیم معاملاتی در MQL ساخته نمی‌شود. تمام محاسبات در Python انجام می‌شود و همه پارامترهای labeling از JSON config خوانده می‌شوند.


## خروجی strict ML-ready

از نسخه `v0.1.3` به بعد، فایل‌های `data/ml_ready/*_ml_ready_*.csv` تنها خروجی مناسب آموزش XGBoost هستند.

این فایل‌ها فقط شامل موارد زیر هستند:

```text
100 configured model features
label
```

در این فایل‌ها ستون‌های زیر عمدا وجود ندارند:

```text
date, symbol, OHLC, raw DXY, entry/exit prices, realized_pips, event_type, outcome_label
```

فایل‌های `data/labels/*_labels_*.csv` فقط برای audit و sanity check هستند.  
فایل‌های `data/label_metadata/*_metadata_*.csv` برای تحلیل زمانی، walk-forward/CPCV و debug نگهداری می‌شوند، اما نباید مستقیما به مدل داده شوند.
