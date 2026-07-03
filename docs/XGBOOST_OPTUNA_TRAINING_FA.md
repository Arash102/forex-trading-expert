# آموزش و ارزیابی XGBoost + Optuna

این مرحله فقط مدل پیش بینی می سازد؛ هنوز مدل معاملاتی نهایی نیست.

## ورودی ها

ورودی مدل فقط فایل های زیر است:

```text
data/ml_ready/*_ml_ready_*.csv
```

این فایل ها باید فقط شامل موارد زیر باشند:

```text
100 configured model features
label
```

برای split زمانی و audit از metadata استفاده می کنیم:

```text
data/label_metadata/*_metadata_*.csv
```

metadata وارد XGBoost نمی شود.

## Missing values

در config پیش فرض:

```json
"missing_values": {
  "strategy": "xgboost_native",
  "dropna": false
}
```

یعنی NaN ها حذف نمی شوند. XGBoost به صورت native با NaN کار می کند و مسیر missing را در هر split یاد می گیرد. این برای فیچرهای سشنی ما مهم است، چون بخشی از NaN ها ساختاری و causal هستند، نه خطای داده.

## Validation

دو روش پشتیبانی می شود:

```text
walk_forward
cpcv
```

برای شروع، walk-forward فعال است. CPCV نیز با group های زمانی، purge event overlap، و embargo پیاده سازی شده است.

## Metrics

متریک های ML شامل این موارد هستند:

```text
accuracy
precision
recall
specificity
f1
balanced_accuracy
mcc
roc_auc
average_precision
log_loss
```

مهم: MCC اضافه شده و در گزارش fold و summary ذخیره می شود.

## اجرای مرحله

ابتدا config محلی بسازید:

```bash
cp configs/ml_config.example.json configs/ml_config.local.json
```

قبل از train، sanity check:

```bash
python scripts/04b_sanity_check_ml.py --config configs/ml_config.local.json
```

سپس train:

```bash
python scripts/04_train_xgb.py --config configs/ml_config.local.json
```

بعد دوباره sanity check بگیرید تا خلاصه نتایج را ببینید:

```bash
python scripts/04b_sanity_check_ml.py --config configs/ml_config.local.json
```

## خروجی ها

```text
data/ml_results/<experiment>/<job>/predictions.csv
data/ml_results/<experiment>/<job>/fold_metrics.csv
data/ml_results/<experiment>/<job>/metrics_summary.csv
data/ml_results/<experiment>/<job>/best_params_by_fold.csv
data/ml_results/<experiment>/run_summary.csv
```

این خروجی ها generated هستند و وارد Git نمی شوند.

## نکته معاملاتی

این مرحله فقط احتمال موفقیت target را می دهد:

```text
p_long
p_short
```

تصمیم معاملاتی بعدا در لایه جدا ساخته می شود:

```text
long اگر p_long بالا و p_short پایین باشد
short اگر p_short بالا و p_long پایین باشد
no trade در حالت conflict یا عدم اطمینان
```
