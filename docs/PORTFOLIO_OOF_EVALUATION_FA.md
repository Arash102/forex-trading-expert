# ارزیابی پورتفوی OOF با Risk-Policy Sweep

این مرحله برای ارزیابی تجمیعی چند setup روی خروجی های `OOF trading evaluation` ساخته شده است. هدف این است که سیستم نهایی فقط بر اساس یک setup قضاوت نشود و اثر ترکیب چند setup، کنترل ریسک، همپوشانی معاملات و انتخاب سیگنال های هم زمان بررسی شود.

## ورودی

برای هر component، فایل زیر خوانده می شود:

```text
 data/trading_eval/<experiment>/<job>/selected_trades.csv
```

هر component با این مشخصات در config تعریف می شود:

```json
{
  "component_id": "eur_fast_long_top2",
  "experiment": "xgb_v0_1_7_rule_context_rule_context_v1",
  "job": "EURUSD_fast_15_8_h16_long",
  "policy": "top_percentile_by_fold",
  "probability_column": "y_prob_raw",
  "top_percentile": 2.0,
  "priority": 1
}
```

## Risk policy های جدید

این نسخه به جای یک کنترل ریسک ثابت، چند سناریو را sweep می کند:

| policy | هدف |
|---|---|
| `no_controls_diagnostic` | سقف خام پتانسیل، فقط برای تشخیص؛ مناسب live نیست |
| `user_max4sym_open8` | قانون اصلی کاربر: حداکثر ۴ معامله برای هر نماد در روز و حداکثر ۸ معامله باز همزمان |
| `non_overlap_same_symbol` | معامله جدید روی همان نماد فقط وقتی پذیرفته می شود که معامله قبلی همان نماد بسته شده باشد |
| `max2_symbol_day` | نسخه محافظه کارانه: حداکثر ۲ معامله برای هر نماد در روز |
| `session_cap` | محدودیت سشن: حداکثر ۱ معامله برای هر نماد در هر سشن |
| `daily_loss_guard` | شبیه baseline کاربر، ولی بعد از ۲ ضرر بسته شده در روز معامله جدید نمی گیرد |

با ریسک ۲٪ برای هر معامله، policy اصلی کاربر این محدودیت ها را دارد:

```text
risk_per_trade = 2%
max_open_trades = 8
max_open_risk = 16%
max_trades_per_symbol_per_day = 4
max_trades_per_day = 8
```

## انتخاب سیگنال ها

ارزیاب پورتفوی زمان مند است و از آینده استفاده نمی کند. معاملات به ترتیب `entry_date` بررسی می شوند. فقط وقتی چند سیگنال در یک زمان دقیق وجود دارد، انتخاب بر اساس `component_priority` و سپس `portfolio_rank_score` انجام می شود. این باعث می شود حذف duplicate یا conflict در یک timestamp، rank-aware باشد، ولی سیگنال های بعدی روز برای انتخاب سیگنال قبلی استفاده نشوند.

## خروجی ها

برای هر پورتفوی و هر risk policy، خروجی ها در این مسیر ساخته می شوند:

```text
data/portfolio_eval/<portfolio>/<risk_policy>/
```

فایل ها:

| فایل | توضیح |
|---|---|
| `portfolio_summary.csv` | خلاصه کلی win rate، payoff، PF، DD، RoR، fold stability |
| `portfolio_selected_trades.csv` | معاملات پذیرفته شده بعد از کنترل ریسک |
| `portfolio_candidate_trades_before_controls.csv` | همه معاملات کاندید قبل از کنترل ریسک |
| `portfolio_risk_control_audit.csv` | علت پذیرش یا رد هر معامله |
| `portfolio_risk_control_reject_summary.csv` | خلاصه علت های رد و سود/ضرر معاملات رد شده |
| `portfolio_fold_metrics.csv` | متریک های fold به fold |
| `portfolio_component_contribution.csv` | سهم هر setup/component از نتیجه پورتفوی |

خلاصه همه حالت ها در این فایل ساخته می شود:

```text
data/portfolio_eval/all_portfolio_summary.csv
```

## اجرای مرحله

```bash
python scripts/07_portfolio_oof_eval.py --config configs/ml_config.local.json
python scripts/07b_compare_portfolios.py --config configs/ml_config.local.json --min-trades 20 --max-risk-of-ruin 0.01 --min-positive-folds 6
```

برای اجرای یک پورتفوی یا policy خاص:

```bash
python scripts/07_portfolio_oof_eval.py --config configs/ml_config.local.json --portfolio p03_mixed_eur_fast_xau_active
python scripts/07_portfolio_oof_eval.py --config configs/ml_config.local.json --risk-policy user_max4sym_open8
```

## معیار تصمیم

برای کاندید پورتفوی قابل ادامه:

```text
profit_factor >= 1.5
risk_of_ruin_dd_25pct <= 1%
positive_folds >= 6 از 9
max_drawdown_pct قابل تحمل
trade_count کافی
```

`no_controls_diagnostic` فقط برای فهمیدن سقف edge است. اگر فقط این policy پاس شود ولی policy های live-feasible پاس نشوند، یعنی مشکل در طراحی کنترل ریسک یا timing/overlap است، نه اینکه سیستم آماده live باشد.

## اصلاح معیارهای پورتفوی ترکیبی بر اساس R

برای پورتفوی هایی که چند نماد دارند، به ویژه ترکیب `EURUSD` و `XAUUSD`، معیارهای مبتنی بر pip برای تصمیم نهایی کافی نیستند. یک pip در EURUSD و یک pip در XAUUSD از نظر ارزش ریسک یکسان نیست. بنابراین در مقایسه پورتفوی، ستون های زیر معیار اصلی هستند:

```text
profit_factor_R
payoff_ratio_R
gross_profit_R
gross_loss_R
expectancy_R
net_R
max_drawdown_R
max_drawdown_pct
risk_of_ruin_dd_25pct
```

ستون های `profit_factor` و `payoff_ratio` هنوز برای بررسی نمادهای منفرد مفیدند، اما در پورتفوی چند نمادی، گزارش مقایسه از `profit_factor_R` استفاده می کند.

## الزام پوشش دو طرف بازار

هدف نهایی سیستم این نیست که فقط یک جهت را معامله کند. برای هر نماد باید دست کم یک setup خرید و یک setup فروش قابل بررسی داشته باشیم:

```text
EURUSD buy
EURUSD sell
XAUUSD buy
XAUUSD sell
```

در config، چند template برای این کار اضافه شده است:

```text
p06_eurusd_buy_sell_probe
p07_xau_buy_sell_core
p08_side_complete_core_optional
p09_eur_short_probe_optional
p10_eur_short_top5_probe_optional
```

در حال حاضر، اگر خروجی آموزشی/OOF برای EURUSD sell هنوز ساخته نشده باشد، کامپوننت های مربوط به EURUSD short با `required=false` تعریف شده اند تا اجرای پورتفوی fail نشود. اما برای پذیرش نهایی live، کامپوننت های sell باید واقعا train/evaluate شوند و `loaded_component_count` باید با `configured_component_count` برابر شود.

برای تکمیل EURUSD sell، ابتدا job های short را train و بعد OOF trading evaluation را دوباره اجرا کنید:

```bash
python scripts/05_train_candidate_sets.py --config configs/ml_config.local.json --candidate-set rule_context_v1 --job EURUSD_fast_15_8_h16_short
python scripts/05_train_candidate_sets.py --config configs/ml_config.local.json --candidate-set rule_context_v1 --job EURUSD_runner_20_10_h24_short
python scripts/06_oof_trading_eval.py --config configs/ml_config.local.json --experiment xgb_v0_1_7_rule_context_rule_context_v1
python scripts/07_portfolio_oof_eval.py --config configs/ml_config.local.json
```

برای تصمیم نهایی، هیچ پورتفوی side-complete نباید فقط به خاطر optional بودن یک سمت ناقص پذیرفته شود. شرط کنترلی این است:

```text
loaded_component_count == configured_component_count
```
