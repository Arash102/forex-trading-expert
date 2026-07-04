# Rule-inspired meta-labeling v0.1.7

این مرحله candidate های عمومی را کنار نمی گذارد، اما candidate اصلی را از setup های واقعی research می سازد.

Preset اصلی:

```text
rule_core_v1
```

این preset بر اساس symbol و side مسیر می دهد:

```text
EURUSD long  -> AH_ATR2 + L3_R52 + L4_NOT2
EURUSD short -> late NY Tue/Fri + late NY ATR high + London weak
XAUUSD long  -> XAU_L_ASIAL_REJECT + XAU_L_H1UP
XAUUSD short -> XAU_DC_NOFRI
```

قواعد فقط از feature های همان کندل استفاده می کنند و از label، realized_pips، event_type، entry/exit و آینده استفاده نمی کنند.

اجرای پیشنهادی:

```bash
python -m pytest tests/test_ml_training.py
python scripts/05a_sanity_check_candidates.py --config configs/ml_config.local.json --save
```

بعد train هدفمند:

```bash
python scripts/05_train_candidate_sets.py --config configs/ml_config.local.json --candidate-set rule_core_v1 --job EURUSD_fast_15_8_h16_short
python scripts/05_train_candidate_sets.py --config configs/ml_config.local.json --candidate-set rule_core_v1 --job XAUUSD_runner_2200_1100_h40_long
python scripts/05_train_candidate_sets.py --config configs/ml_config.local.json --candidate-set rule_core_v1 --job XAUUSD_active_1500_800_h32_short
```

اگر candidate ها خیلی کم شدند، اول positive_rate و fold_count را از sanity بررسی کن و بعد train کامل بگیر.


## Rule context v1

`rule_core_v1` is intentionally strict and should be treated as raw rule evidence, not as an XGBoost training universe when it produces too few rows.  
`rule_context_v1` is the broader meta-labeling universe: it keeps the same directional setup logic but relaxes the exact entry triggers so the meta model has enough candidate rows per walk-forward fold.

Recommended order:

1. Run candidate sanity.
2. Train `rule_context_v1` first.
3. Keep `rule_core_v1` for raw rule/trading audit, not for XGBoost unless it has enough rows.
