# کاهش Risk of Ruin برای پورتفوی setup-inventory

این مرحله بعد از `09_inventory_portfolio_eval` اجرا می‌شود و منطق ورود یا خروج را تغییر نمی‌دهد. هدف آن بررسی این است که با چه سطح ریسک ثابت به ازای هر معامله و با چه وزن‌دهی ساده‌ای می‌توان `risk_of_ruin` را پایین آورد، بدون اینکه ساختار ۱۲ setup از بین برود.

## خروجی‌ها

مسیر پیش‌فرض:

```text
data/inventory_ror_optimization/
```

فایل‌ها:

```text
ror_reduction_summary.csv
ror_reduction_pass.csv
recommended_risk_plan.csv
ror_component_stress_summary.csv
```

## منطق

هر معامله قبلا با `pnl_R` ذخیره شده است. این مرحله مسیر سود و زیان را با این رابطه بازسازی می‌کند:

```text
trade_return_pct = pnl_R * risk_per_trade_pct * risk_multiplier
```

بنابراین:

- `risk_per_trade_pct` اندازه ریسک پایه هر معامله است.
- `risk_multiplier` می‌تواند برای نماد، سمت یا setup خاص کمتر از ۱ شود.
- معیارهای `PF_R`، `max_drawdown_pct` و `risk_of_ruin` دوباره محاسبه می‌شوند.

## اجرای اصلی

```powershell
python scripts/10_inventory_ror_optimizer.py --config configs/ml_config.local.json
```

برای فقط پورتفوی ۱۲ setup:

```powershell
python scripts/10_inventory_ror_optimizer.py --config configs/ml_config.local.json --portfolio ip01_core_12_side_complete
```

برای فقط یک risk policy:

```powershell
python scripts/10_inventory_ror_optimizer.py --config configs/ml_config.local.json --portfolio ip01_core_12_side_complete --risk-policy daily_loss_guard
```

## معیار پاس

پیش‌فرض:

```text
trade_count >= 60
profit_factor_R >= 1.50
risk_of_ruin_dd_25pct <= 1%
positive_folds >= 6
loaded_component_count == configured_component_count
```

این مرحله برای انتخاب sizing مناسب است، نه برای اثبات نهایی live-ready بودن سیستم.
