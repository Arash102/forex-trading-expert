# حافظه مرجع پروژه DebCo

این فایل برای یادآوری تصمیم های قطعی پروژه است.

## قوانین غیرقابل نقض

1. Python تنها منبع حقیقت است.
2. هیچ فیچر، regime، DXY، GMMA یا rule داخل MQL بازسازی یا تقریبی نمی شود.
3. MQL فقط برای dashboard، screenshot و نمایش وضعیت مجاز است.
4. هر نسخه باید در Git commit شود.
5. configها باید JSON باشند.
6. DXY آماده از بروکر لازم نیست؛ DXY داخل Python از اجزا ساخته می شود.

## فرمول DXY پروژه

```text
DXY = 50.14348112
× EURUSD^(-0.576)
× USDJPY^(0.136)
× GBPUSD^(-0.119)
× USDCAD^(0.091)
× USDSEK^(0.042)
× USDCHF^(0.036)
```

بعد:

```text
dxy_inverse_close = 100 / dxy_close
index_close = dxy_inverse_close
```

## فاز جدید

هدف فعلی ساخت پروژه تمیز با مراحل زیر است:

1. دریافت کندل خام از MT5 با config
2. ساخت DXY برای همان بازه
3. ذخیره raw dataset
4. ساخت feature dataset
5. ML با XGBoost + Optuna
6. validation با CPCV یا walk-forward
7. triple barrier labels
8. trading model/backtest
9. live execution فقط از Python
