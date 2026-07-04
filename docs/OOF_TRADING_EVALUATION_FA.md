# OOF Trading Evaluation

هدف این مرحله این است که پیش بینی های out-of-fold مدل را مثل سیگنال معاملاتی واقعی ارزیابی کنیم. این مرحله جایگزین متریک های ML نیست، بلکه تصمیم نهایی معاملاتی را با معیارهای زیر می سنجد:

- win rate
- payoff ratio
- profit factor
- expectancy
- max drawdown
- drawdown duration
- risk of ruin
- trades per month

## فرض سرمایه و ریسک

پیش فرض فعلی:

```text
initial_capital = 1000 USD
risk_per_trade = 2%
```

یعنی هر معامله با ریسک 1R برابر 20 دلار ارزیابی می شود. اگر نتیجه معامله +2R باشد، سود آن حدود 40 دلار است. اگر نتیجه -1R باشد، ضرر آن حدود 20 دلار است.

## حالت های ارزیابی

### fixed_threshold

برای هر probability column و threshold، هر ردیفی که احتمال آن از threshold بالاتر باشد معامله فرضی می شود.

### top_percentile_by_fold

برای بررسی high-confidence zone، در هر fold فقط top درصدهای بالای probability انتخاب می شوند. این حالت برای تشخیص قدرت ranking مدل است و مستقیما قانون live نیست مگر بعدا به threshold قابل اجرا تبدیل شود.

### rolling_oof_target_precision

برای هر fold، threshold از fold های قبلی انتخاب می شود و روی fold بعدی اعمال می شود. این روش خوش بینانه تر از انتخاب threshold روی کل OOF نیست و برای بررسی هدف win rate/precision بالا مفید است.

## فایل های خروجی

برای هر job:

```text
fixed_threshold_summary.csv
top_percentile_summary.csv
rolling_target_precision_summary.csv
trading_fold_metrics.csv
selected_trades.csv
trading_summary.csv
```

خروجی کلی:

```text
data/trading_eval/all_trading_summary.csv
```

## معیار قبولی پیشنهادی

برای سیستم قابل بررسی:

```text
profit_factor >= 1.5
risk_of_ruin_25dd <= 1%
positive trade frequency enough for use
max_drawdown_pct acceptable
```

برای high-confidence mode:

```text
win_rate >= 60%
profit_factor >= 2
trade_count not too small
```
