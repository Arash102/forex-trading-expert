# معماری پروژه DebCo

## اصل معماری

هیچ محاسبه تقریبی داخل MQL نداریم. هر چیزی که در بک تست باعث تصمیم شده باید در Python و با همان pipeline ساخته شود.

```text
MT5 candles + MT5 broker data
        ↓
Python data layer
        ↓
DXY builder from components
        ↓
Feature builder
        ↓
Label builder / ML model / Rule engine
        ↓
Trading engine / Risk manager / Backtester
        ↓
MT5 Python API order execution
        ↓
MQL monitor only: dashboard + screenshots
```

## ماژول ها

### `src/debco/data`

- اتصال به MT5
- دریافت کندل خام برای نمادها
- دریافت اجزای DXY
- ساخت DXY با فرمول رسمی پروژه
- ذخیره CSV خام

### `src/debco/features`

- ساخت همه فیچرهای مشترک EURUSD و XAUUSD
- session features
- ATR/RSI/GMMA
- DXY inverse / index close
- market_x / market_y / market_regime
- Asia/London context
- ZigZag / H1 trend / volatility context

### `src/debco/labels`

- Triple Barrier طبق Lopez de Prado
- TP/SL/horizon قابل تنظیم در JSON
- امکان label برای long/short یا جهت کلی

### `src/debco/validation`

- walk-forward
- CPCV طبق Lopez de Prado
- purge/embargo در نسخه های بعدی

### `src/debco/ml`

- XGBoost
- Optuna
- کنترل search space از JSON
- انتخاب مدل بر اساس metricهای تعریف شده

### `src/debco/trading`

- تبدیل probability/score به signal
- entry/exit simulation
- max trades/day
- min daily opportunity layer در فاز rule_base یا coverage
- risk management

### `src/debco/live`

- Python live router
- فقط آخرین کندل بسته شده
- ارسال order به MT5 Python API
- مدیریت پوزیشن های باز

## تصمیم فعلی پروژه

فاز فعلی: ساخت موتور تمیز data + features + ML research. استراتژی های rule-based قبلی به عنوان benchmark نگه داشته می شوند، اما معماری جدید باید به صورت clean و versioned جلو برود.
