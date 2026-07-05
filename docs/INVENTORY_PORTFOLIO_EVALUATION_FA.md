# ارزیابی پورتفوی مبتنی بر Setup Inventory - v0.1.10

این مرحله بعد از ساخت و ارزیابی setup inventory اجرا می شود. هدف آن این است که ۱۲ setup منتخب، یعنی سه setup برای هر خانه زیر، به صورت یک پورتفوی واقعی و با کنترل ریسک مشترک ارزیابی شوند:

- EURUSD long
- EURUSD short
- XAUUSD long
- XAUUSD short

در این مرحله دیگر صرفا عملکرد جداگانه setup مهم نیست. مسئله اصلی این است که آیا ترکیب setup ها در کنار هم، با محدودیت های ریسک زنده، همچنان سودده و پایدار می ماند یا نه.

## ورودی اصلی

اسکریپت از فایل زیر استفاده می کند:

```text
 data/setup_inventory/setup_best_viable_policy_by_setup.csv
```

این فایل باید بعد از اجرای مراحل زیر ساخته شده باشد:

```bash
python scripts/08b_evaluate_setup_inventory_trading.py --config configs/ml_config.local.json
python scripts/08c_compare_setup_inventory.py --config configs/ml_config.local.json --min-trades 20 --min-pf-r 1.30 --max-ror 0.05
```

برای هر setup، policy منتخب از همین فایل خوانده می شود. سپس اسکریپت معاملات متناظر را از مسیر زیر بارگذاری می کند:

```text
 data/trading_eval/<experiment>/<job>/selected_trades.csv
```

## پورتفوی های ساخته شده

اسکریپت به صورت پیش فرض این پورتفوی ها را می سازد:

| portfolio | توضیح |
|---|---|
| `ip01_core_12_side_complete` | ۱۲ setup اصلی، سه setup برای هر symbol/side |
| `ip02_core_12_plus_xau_sell_dxy_pressure` | ۱۲ setup اصلی به علاوه setup پشتیبان XAU sell DXY pressure |
| `ip03_eurusd_6_setups` | فقط EURUSD، سه buy و سه sell |
| `ip04_xauusd_6_setups` | فقط XAUUSD، سه buy و سه sell |
| `ip05_buy_side_6_setups` | همه setup های خرید EURUSD و XAUUSD |
| `ip06_sell_side_6_setups` | همه setup های فروش EURUSD و XAUUSD |
| `ip07_low_ror_subset` | زیرمجموعه کم ریسک از setup های منتخب، در صورت وجود |

## دستور اجرا

```bash
python scripts/09_inventory_portfolio_eval.py --config configs/ml_config.local.json
```

برای اجرای یک پورتفوی خاص:

```bash
python scripts/09_inventory_portfolio_eval.py --config configs/ml_config.local.json --portfolio ip01_core_12_side_complete
```

برای اجرای یک risk policy خاص:

```bash
python scripts/09_inventory_portfolio_eval.py --config configs/ml_config.local.json --risk-policy session_cap
```

## مقایسه خروجی ها

```bash
python scripts/09b_compare_inventory_portfolios.py --config configs/ml_config.local.json --min-trades 60 --min-pf-r 1.50 --max-ror 0.01 --min-positive-folds 6
```

## خروجی ها

خروجی ها در مسیر زیر ذخیره می شوند:

```text
 data/inventory_portfolio_eval/
```

فایل های مهم:

| فایل | توضیح |
|---|---|
| `all_inventory_portfolio_summary.csv` | خلاصه همه پورتفوی ها و risk policy ها |
| `inventory_portfolio_decision_matrix.csv` | جدول تصمیم نهایی با ستون pass/fail |
| `inventory_portfolio_robust_pass.csv` | فقط پورتفوی هایی که معیار سختگیرانه را پاس کرده اند |
| `portfolio_selected_trades.csv` | معاملات پذیرفته شده هر پورتفوی/سیاست |
| `portfolio_candidate_trades_before_controls.csv` | معاملات candidate قبل از کنترل ریسک |
| `portfolio_risk_control_audit.csv` | audit دلیل پذیرش یا رد هر معامله |
| `portfolio_component_contribution.csv` | سهم هر setup در پورتفوی |

## معیار تصمیم

برای پورتفوی نهایی، معیار اصلی R-based است، نه pip-based:

```text
profit_factor_R >= 1.5
risk_of_ruin_dd_25pct <= 1%
positive_folds >= 6 از 9
loaded_component_count == configured_component_count
```

برای پورتفوی ۱۲ تایی، ستون های پوشش هم مهم هستند:

```text
eurusd_long_setup_count >= 3
eurusd_short_setup_count >= 3
xauusd_long_setup_count >= 3
xauusd_short_setup_count >= 3
```

اگر setup ها جداگانه خوب باشند اما پورتفوی نهایی fail شود، مشکل احتمالا از overlap زمانی، تمرکز ریسک، همزمانی ضررها، یا policy های کنترل ریسک است. در آن حالت باید به جای تغییر کورکورانه setup ها، `portfolio_component_contribution.csv` و `portfolio_risk_control_audit.csv` بررسی شود.
