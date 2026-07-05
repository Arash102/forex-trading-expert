# گزارش نهایی استراتژی و مشخصات اجرای زنده

این مرحله بعد از `v0.1.11` اجرا می شود و هدف آن ساخت یک بسته گزارش گیری نهایی از پورتفوی ۱۲ ستاپی است. این مرحله هیچ مدل جدیدی train نمی کند، قوانین ورود را تغییر نمی دهد و فقط خروجی های زیر را از نتایج موجود می سازد:

- `final_strategy_report_FA.md`
- `final_strategy_summary.csv`
- `risk_mode_comparison.csv`
- `setup_selection_matrix.csv`
- `component_stress_selected.csv`
- `live_execution_spec.json`

## ورودی ها

این script به خروجی های مراحل قبل نیاز دارد:

```text
data/setup_inventory/setup_best_viable_policy_by_setup.csv
data/inventory_portfolio_eval/
data/inventory_ror_optimization/ror_reduction_summary.csv
data/inventory_ror_optimization/ror_component_stress_summary.csv
```

## پورتفوی منتخب پیش فرض

```text
portfolio: ip01_core_12_side_complete
risk_policy: daily_loss_guard
risk_plan: xau_sell_50pct
risk_per_trade: 1.0%
```

این انتخاب برای گزارش متعادل است. حالت ۰.۷۵٪ محافظه کارتر و حالت ۱.۵٪ تهاجمی تر است.

## اجرا

```bash
python scripts/11_generate_final_strategy_report.py --config configs/ml_config.local.json
```

خروجی در مسیر زیر ساخته می شود:

```text
data/final_strategy_report
```

## نکته اجرایی

این خروجی به معنی شروع مستقیم حساب واقعی نیست. مرحله بعد باید forward test/demo باشد. قاعده پروژه همچنان برقرار است: Python تنها منبع تصمیم معاملاتی است و MT5 فقط داده و اجرای سفارش را انجام می دهد.
