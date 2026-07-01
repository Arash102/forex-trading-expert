# DebCo Research Engine

این پروژه نسخه تمیز و ماژولار سیستم تحقیق، بک تست و اجرای دمو برای EURUSD و XAUUSD است.

اصل ثابت پروژه:

```text
Python تنها منبع حقیقت برای محاسبه فیچر، DXY، لیبل، مدل، سیگنال و تصمیم معاملاتی است.
MT5 فقط منبع داده و ابزار اجرای سفارش است.
MQL فقط برای داشبورد، عکس و مانیتور مجاز است و حق تصمیم معاملاتی ندارد.
```

## شروع سریع

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

برای دریافت داده خام از MT5 و ساخت DXY:

```bash
python scripts/01_fetch_data.py --config configs/data_config.example.json
```

برای ساخت فیچرها:

```bash
python scripts/02_build_features.py --data-config configs/data_config.example.json --features-config configs/features_config.example.json
```

برای اجرای تست ML با XGBoost/Optuna:

```bash
python scripts/03_train_xgb.py --config configs/ml_config.example.json
```

## ترتیب فازهای پروژه

1. `data`: دریافت کندل های خام از MT5 و ساخت DXY از اجزای آن
2. `features`: ساخت فیچرهای مشترک EURUSD/XAUUSD
3. `labels`: ساخت label با Triple Barrier طبق Lopez de Prado
4. `validation`: CPCV یا walk-forward
5. `ml`: XGBoost + Optuna
6. `trading`: تبدیل پیش بینی به معامله، SL/TP/position sizing/risk manager/backtest
7. `live`: اجرای دمو با Python و ارسال سفارش به MT5
8. `monitor`: MQL فقط برای داشبورد/عکس، بدون منطق معاملاتی

## قانون مهم Git

هیچ خروجی zip پراکنده مبنای کار نیست. هر تغییر باید commit شود.

پیشنهاد commit اول:

```bash
git init
git add .
git commit -m "chore: bootstrap clean debco research engine"
```
