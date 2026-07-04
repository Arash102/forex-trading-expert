# v0.1.10 - Setup-Specific Edge Inventory and Meta-Labeling

هدف این مرحله ساخت یک inventory کامل از setup های معاملاتی است؛ نه فقط یک candidate filter عمومی. سیستم نهایی باید برای هر نماد و هر سمت چند setup مستقل داشته باشد:

- EURUSD buy: حداقل 3 setup
- EURUSD sell: حداقل 3 setup
- XAUUSD buy: حداقل 3 setup
- XAUUSD sell: حداقل 3 setup

در این مرحله هر setup جداگانه candidate universe می سازد، بعد XGBoost فقط روی همان setup نقش meta-filter دارد. این کار مانع می شود setup های خوب و بد داخل یک rule_context مخلوط شوند.

## Setup های اولیه

در `configs/ml_config.example.json` بخش `setup_inventory` اضافه شده است. نسخه اولیه 12 setup دارد:

### EURUSD buy

1. `EUR_AH_ATR2_BUY`
2. `EUR_L3_R52_BUY`
3. `EUR_L4_NOT2_BUY`

### EURUSD sell

1. `EUR_LATENY_TUEFRI_SHORT`
2. `EUR_LATENY_ATRHI_SHORT`
3. `EUR_LONDON_WEAK_SHORT`

### XAUUSD buy

1. `XAU_ASIAL_REJECT_BUY`
2. `XAU_H1UP_BUY`
3. `XAU_BUY_BREAKOUT_PULLBACK`

### XAUUSD sell

1. `XAU_DC_NOFRI_SHORT`
2. `XAU_ACTIVE_BREAKDOWN_SHORT`
3. `XAU_SHORT_REVERSAL`

بعضی setup ها برگرفته از playbook قبلی هستند و بعضی ها discovery context هستند. هیچ کدام تا قبل از OOF trading evaluation نباید نهایی فرض شوند.

## اجرای پیشنهادی

اول sanity بگیر:

```bash
python scripts/08a_sanity_check_setup_inventory.py --config configs/ml_config.local.json --save
```

اگر خواستی فقط یک سمت را بررسی کنی:

```bash
python scripts/08a_sanity_check_setup_inventory.py --config configs/ml_config.local.json --symbol EURUSD --side short --save
```

بعد train مرحله ای بگیر. برای شروع بهتر است همه را یکجا اجرا نکنی و اول سمت های ضعیف را هدف بگیری:

```bash
python scripts/08_train_setup_inventory.py --config configs/ml_config.local.json --symbol EURUSD --side short
```

یا یک setup مشخص:

```bash
python scripts/08_train_setup_inventory.py --config configs/ml_config.local.json --setup-id EUR_LONDON_WEAK_SHORT
```

بعد trading evaluation setup ها:

```bash
python scripts/08b_evaluate_setup_inventory_trading.py --config configs/ml_config.local.json
python scripts/08c_compare_setup_inventory.py --config configs/ml_config.local.json --min-trades 20 --min-pf-r 1.30 --max-ror 0.05
```

## خروجی ها

در `data/setup_inventory` این فایل ها ساخته می شوند:

- `setup_raw_audit.csv`
- `setup_coverage_matrix.csv`
- `setup_training_summary.csv`
- `setup_trading_eval_summary.csv`
- `setup_best_policy_by_setup.csv`
- `setup_trading_coverage_matrix.csv`
- `setup_inventory_decision_matrix.csv`
- `setup_inventory_coverage_decision.csv`

## معیار انتخاب setup

برای ورود به پورتفوی آزمایشی، یک setup بهتر است حداقل این شرایط را داشته باشد:

- `trade_count >= 20`
- `profit_factor_R >= 1.30` به عنوان soft pass
- ترجیحا `profit_factor_R >= 1.50` برای strong pass
- `risk_of_ruin_dd_25pct <= 5%` برای soft pass
- ترجیحا `risk_of_ruin_dd_25pct <= 1%` برای strong pass
- مثبت بودن چند fold، نه فقط سود aggregate

هدف نهایی این است که برای هر خانه از ماتریس symbol/side حداقل سه setup قابل دفاع داشته باشیم، سپس دوباره portfolio OOF evaluation را با مجموعه منتخب اجرا کنیم.


## v0.1.10 sanity tuning note

If a setup-specific candidate has `fold_count_after = 0`, it is too sparse for walk-forward meta-labeling. The first sanity pass showed several XAU discovery setups were too narrow. The updated defaults intentionally widen the setup context for the sparse XAU discovery candidates and slightly loosen `EUR_LATENY_TUEFRI_SHORT` so each setup can be evaluated across more folds before training. These wider contexts are not final trading rules; they are candidate universes for meta-labeling.

## اصلاح طراحی: Repair + Redesign

در این نسخه، وقتی یک setup در sanity خروجی ضعیف، sparse یا fold_count پایین داشته باشد، فقط همان فیلتر بازتر نمی شود. یک دسته setup جایگزین با فرضیه معاملاتی متفاوت نیز وارد inventory می شود تا بتوانیم بین «بهبود همان ایده» و «جایگزینی با edge متفاوت» تصمیم بگیریم.

setup های redesign اضافه شده:

- EUR_BUY_MOMENTUM_OVERLAP: خرید مومنتوم در overlap بعد از reclaim سقف آسیا.
- EUR_BUY_REL_STRENGTH: خرید بر اساس relative strength و market_x/market_y.
- EUR_SELL_LONDON_BREAKDOWN: فروش شکست/ضعف لندن، مستقل از late-NY exhaustion.
- EUR_SELL_H1DOWN_CONT: فروش ادامه روند نزولی H1.
- XAU_BUY_MOMENTUM_CONT: خرید ادامه مومنتوم با H1 غیرمنفی.
- XAU_BUY_DXY_TREND: خرید XAU با زمینه DXY inverse حمایتی.
- XAU_SELL_ASIA_LOW_FAIL: فروش شکست یا failure کف آسیا.
- XAU_SELL_H1DOWN_CONT: فروش ادامه روند نزولی H1.

معیار تصمیم بعد از sanity:

1. اگر نسخه اصلی setup rows/folds کافی و lift مناسب داشت، برای train باقی می ماند.
2. اگر نسخه اصلی sparse یا lift ضعیف بود، redesign های همان symbol/side با آن مقایسه می شوند.
3. در مرحله train، هدف انتخاب ۳ setup نهایی برای هر symbol/side است، نه اجبار به نگه داشتن setup های اولیه.
