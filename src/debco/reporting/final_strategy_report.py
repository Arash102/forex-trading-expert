from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd



REQUIRED_SUMMARY_COLUMNS = {
    "portfolio",
    "risk_policy",
    "risk_plan",
    "risk_per_trade_pct",
    "trade_count",
    "profit_factor_R",
    "net_R_weighted",
    "risk_of_ruin_dd_25pct",
    "positive_folds",
}


def _read_csv(path: str | Path, *, required: bool = True) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {p}")
        return pd.DataFrame()
    return pd.read_csv(p)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _is_missing(value: Any) -> bool:
    if isinstance(value, (list, dict, tuple)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _sanitize_json_value(value: Any, digits: int = 6) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _sanitize_json_value(v, digits=digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(v, digits=digits) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_json_value(v, digits=digits) for v in value]
    if isinstance(value, (np.floating, float)):
        out = float(value)
        if not np.isfinite(out):
            return None
        return round(out, digits)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _round_dict_values(d: Mapping[str, Any], digits: int = 6) -> dict[str, Any]:
    return {str(k): _sanitize_json_value(v, digits=digits) for k, v in d.items()}


def _fmt_pct(value: Any, digits: int = 2) -> str:
    return f"{100.0 * _to_float(value):.{digits}f}%"


def _fmt_money(value: Any, digits: int = 2) -> str:
    return f"${_to_float(value):,.{digits}f}"


def _fmt_num(value: Any, digits: int = 3) -> str:
    return f"{_to_float(value):,.{digits}f}"


def validate_ror_summary(df: pd.DataFrame) -> None:
    missing = REQUIRED_SUMMARY_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"ROR summary missing required columns: {sorted(missing)}")


def select_strategy_row(summary: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    validate_ror_summary(summary)
    df = summary.copy()
    for col in ["risk_per_trade_pct", "trade_count", "profit_factor_R", "risk_of_ruin_dd_25pct", "positive_folds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    portfolio = str(cfg.get("selected_portfolio", "ip01_core_12_side_complete"))
    risk_policy = str(cfg.get("selected_risk_policy", "daily_loss_guard"))
    risk_plan = str(cfg.get("selected_risk_plan", "xau_sell_50pct"))
    risk_per_trade = float(cfg.get("selected_risk_per_trade", 0.01))

    exact = df[
        (df["portfolio"].astype(str) == portfolio)
        & (df["risk_policy"].astype(str) == risk_policy)
        & (df["risk_plan"].astype(str) == risk_plan)
        & (np.isclose(df["risk_per_trade_pct"], risk_per_trade, atol=1e-10, rtol=0.0))
    ]
    if not exact.empty:
        return exact.iloc[0]

    fallback = df[
        (df["portfolio"].astype(str) == portfolio)
        & (df["risk_policy"].astype(str) == risk_policy)
        & (df["risk_plan"].astype(str) == risk_plan)
    ].copy()
    if fallback.empty:
        fallback = df.copy()
    if "portfolio_pass" in fallback.columns:
        fallback["portfolio_pass_sort"] = fallback["portfolio_pass"].map(_boolish)
    else:
        fallback["portfolio_pass_sort"] = False
    fallback = fallback.sort_values(
        ["portfolio_pass_sort", "risk_of_ruin_dd_25pct", "profit_factor_R", "net_R_weighted"],
        ascending=[False, True, False, False],
    )
    return fallback.iloc[0]


def select_comparison_rows(summary: pd.DataFrame, selected: Mapping[str, Any], cfg: Mapping[str, Any]) -> pd.DataFrame:
    risks = [float(x) for x in cfg.get("comparison_risk_per_trades", [0.0075, 0.01, 0.015])]
    df = summary.copy()
    df["risk_per_trade_pct"] = pd.to_numeric(df["risk_per_trade_pct"], errors="coerce")
    mask = (
        df["portfolio"].astype(str).eq(str(selected.get("portfolio")))
        & df["risk_policy"].astype(str).eq(str(selected.get("risk_policy")))
        & df["risk_plan"].astype(str).eq(str(selected.get("risk_plan")))
        & df["risk_per_trade_pct"].apply(lambda x: any(np.isclose(x, r, atol=1e-10, rtol=0.0) for r in risks))
    )
    cols = [
        "portfolio",
        "risk_policy",
        "risk_plan",
        "risk_per_trade_pct",
        "trade_count",
        "win_rate",
        "payoff_ratio_R",
        "profit_factor_R",
        "net_R_weighted",
        "net_return_pct_on_initial",
        "net_dollars",
        "max_drawdown_pct",
        "max_drawdown_dollars",
        "risk_of_ruin_dd_25pct",
        "risk_of_ruin_dd_30pct",
        "positive_folds",
        "worst_fold_net_R_weighted",
    ]
    out = df.loc[mask, [c for c in cols if c in df.columns]].copy()
    return out.sort_values("risk_per_trade_pct").reset_index(drop=True)


def load_selected_setups(best_policy_path: str | Path, core_setup_ids: list[str]) -> pd.DataFrame:
    df = _read_csv(best_policy_path, required=True)
    if "setup_id" not in df.columns:
        raise ValueError("setup best-policy file must contain setup_id")
    order = {sid: i for i, sid in enumerate(core_setup_ids)}
    out = df[df["setup_id"].astype(str).isin(core_setup_ids)].copy()
    out["setup_order"] = out["setup_id"].astype(str).map(order).fillna(999).astype(int)
    keep = [
        "setup_order",
        "setup_id",
        "family",
        "symbol",
        "side",
        "job",
        "experiment",
        "policy",
        "probability_column",
        "threshold",
        "top_percentile",
        "trade_count",
        "win_rate",
        "payoff_ratio_R",
        "profit_factor_R",
        "net_R",
        "risk_of_ruin_dd_25pct",
        "positive_folds",
        "candidate_pass_soft",
    ]
    return out[[c for c in keep if c in out.columns]].sort_values("setup_order").reset_index(drop=True)


def selected_component_stress(component_path: str | Path, selected: Mapping[str, Any]) -> pd.DataFrame:
    df = _read_csv(component_path, required=False)
    if df.empty:
        return df
    mask = (
        df["portfolio"].astype(str).eq(str(selected.get("portfolio")))
        & df["risk_policy"].astype(str).eq(str(selected.get("risk_policy")))
        & df["risk_plan"].astype(str).eq(str(selected.get("risk_plan")))
    )
    out = df.loc[mask].copy()
    sort_cols = [c for c in ["net_R_weighted", "profit_factor_R_weighted"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return out.reset_index(drop=True)


def risk_policy_controls(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    pe = config.get("portfolio_eval", {})
    base = dict(pe.get("risk_controls", {}) or {})
    sweep = pe.get("risk_policy_sweep", {}) or {}
    for p in sweep.get("policies", []):
        if str(p.get("name")) == str(name):
            merged = dict(base)
            merged.update(p.get("controls", {}) or {})
            return merged
    return base


def risk_plan(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    roc = config.get("inventory_ror_optimizer", {})
    for p in roc.get("risk_plans", []):
        if str(p.get("name")) == str(name):
            return dict(p)
    return {"name": name, "description": "Selected risk plan was not found in config."}


def acceptance_status(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, bool]:
    criteria = config.get("final_strategy_report", {}).get("acceptance_criteria", {})
    out = {
        "profit_factor_R_ok": _to_float(row.get("profit_factor_R")) >= float(criteria.get("min_profit_factor_R", 1.5)),
        "risk_of_ruin_dd_25pct_ok": _to_float(row.get("risk_of_ruin_dd_25pct"), 1.0) <= float(criteria.get("max_risk_of_ruin_dd_25pct", 0.01)),
        "positive_folds_ok": _to_float(row.get("positive_folds")) >= float(criteria.get("min_positive_folds", 6)),
    }
    if bool(criteria.get("require_all_components_loaded", True)):
        out["all_components_loaded_ok"] = _to_float(row.get("loaded_component_count")) == _to_float(row.get("configured_component_count"))
    if bool(criteria.get("require_side_complete_3x3", True)):
        out["side_complete_3x3_ok"] = _boolish(row.get("side_complete_3x3_configured"))
    out["overall_pass"] = all(out.values())
    return out


def build_live_execution_spec(
    *,
    config: Mapping[str, Any],
    selected: Mapping[str, Any],
    setups: pd.DataFrame,
) -> dict[str, Any]:
    fcfg = config.get("final_strategy_report", {})
    return {
        "strategy_id": str(fcfg.get("strategy_id", "debco_final_strategy")),
        "source_milestones": [
            "v0.1.10-setup-inventory-portfolio-ok",
            "v0.1.11-inventory-ror-optimization-ok",
        ],
        "portfolio": str(selected.get("portfolio")),
        "risk_policy": str(selected.get("risk_policy")),
        "risk_plan": str(selected.get("risk_plan")),
        "risk_per_trade": _to_float(selected.get("risk_per_trade_pct")),
        "initial_capital_reference": _to_float(selected.get("initial_capital"), _to_float(fcfg.get("initial_capital"), 1000.0)),
        "metrics": _round_dict_values(
            {
                "trade_count": selected.get("trade_count"),
                "win_rate": selected.get("win_rate"),
                "payoff_ratio_R": selected.get("payoff_ratio_R"),
                "profit_factor_R": selected.get("profit_factor_R"),
                "net_R_weighted": selected.get("net_R_weighted"),
                "net_dollars": selected.get("net_dollars"),
                "max_drawdown_pct": selected.get("max_drawdown_pct"),
                "risk_of_ruin_dd_25pct": selected.get("risk_of_ruin_dd_25pct"),
                "positive_folds": selected.get("positive_folds"),
                "worst_fold_net_R_weighted": selected.get("worst_fold_net_R_weighted"),
            }
        ),
        "risk_policy_controls": risk_policy_controls(config, str(selected.get("risk_policy"))),
        "risk_plan_weights": risk_plan(config, str(selected.get("risk_plan"))),
        "selected_setups": _sanitize_json_value(setups.to_dict("records")),
        "acceptance_status": acceptance_status(selected, config),
        "forward_test_recommendation": str(fcfg.get("forward_test_recommendation", "Paper/demo forward test first.")),
    }


def render_markdown(
    *,
    selected: Mapping[str, Any],
    comparison: pd.DataFrame,
    setups: pd.DataFrame,
    components: pd.DataFrame,
    spec: Mapping[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# گزارش نهایی استراتژی ۱۲ ستاپی")
    lines.append("")
    lines.append("این گزارش فقط بر اساس خروجی های OOF، پورتفوی inventory و بهینه سازی ROR ساخته شده است. این خروجی مجوز ورود مستقیم به حساب واقعی نیست؛ مرحله بعد باید forward test یا demo باشد.")
    lines.append("")
    lines.append("## انتخاب نهایی")
    lines.append("")
    lines.append(f"- Portfolio: `{selected.get('portfolio')}`")
    lines.append(f"- Risk policy: `{selected.get('risk_policy')}`")
    lines.append(f"- Risk plan: `{selected.get('risk_plan')}`")
    lines.append(f"- Risk per trade: `{_fmt_pct(selected.get('risk_per_trade_pct'))}`")
    lines.append(f"- Initial capital reference: `{_fmt_money(selected.get('initial_capital'))}`")
    lines.append("")
    lines.append("## معیارهای اصلی")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Trade count | {_fmt_num(selected.get('trade_count'), 0)} |")
    lines.append(f"| Win rate | {_fmt_pct(selected.get('win_rate'))} |")
    lines.append(f"| Payoff ratio R | {_fmt_num(selected.get('payoff_ratio_R'))} |")
    lines.append(f"| Profit factor R | {_fmt_num(selected.get('profit_factor_R'))} |")
    lines.append(f"| Net R weighted | {_fmt_num(selected.get('net_R_weighted'))}R |")
    lines.append(f"| Net dollars | {_fmt_money(selected.get('net_dollars'))} |")
    lines.append(f"| Max drawdown | {_fmt_pct(selected.get('max_drawdown_pct'))} |")
    lines.append(f"| Risk of 25% drawdown | {_fmt_pct(selected.get('risk_of_ruin_dd_25pct'))} |")
    lines.append(f"| Positive folds | {_fmt_num(selected.get('positive_folds'), 0)} / {_fmt_num(selected.get('folds_with_trades'), 0)} |")
    lines.append(f"| Worst fold net R weighted | {_fmt_num(selected.get('worst_fold_net_R_weighted'))}R |")
    lines.append("")
    status = spec.get("acceptance_status", {})
    lines.append("## وضعیت پذیرش")
    lines.append("")
    for k, v in status.items():
        lines.append(f"- `{k}`: {'PASS' if bool(v) else 'FAIL'}")
    lines.append("")
    if not comparison.empty:
        lines.append("## مقایسه حالت های ریسک")
        lines.append("")
        lines.append("| Risk/trade | Net $ | Max DD | RoR 25% | PF_R |")
        lines.append("|---:|---:|---:|---:|---:|")
        for _, r in comparison.iterrows():
            lines.append(
                f"| {_fmt_pct(r.get('risk_per_trade_pct'))} | {_fmt_money(r.get('net_dollars'))} | {_fmt_pct(r.get('max_drawdown_pct'))} | {_fmt_pct(r.get('risk_of_ruin_dd_25pct'))} | {_fmt_num(r.get('profit_factor_R'))} |"
            )
        lines.append("")
    lines.append("## ماتریس ۱۲ ستاپ")
    lines.append("")
    lines.append("| Setup | Symbol | Side | Policy | Trades | PF_R | RoR 25% |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    for _, r in setups.iterrows():
        policy = str(r.get("policy", ""))
        if pd.notna(r.get("top_percentile", np.nan)):
            policy += f" / top {r.get('top_percentile')}%"
        elif pd.notna(r.get("threshold", np.nan)):
            policy += f" / threshold {r.get('threshold')}"
        lines.append(
            f"| `{r.get('setup_id')}` | {r.get('symbol', '')} | {r.get('side', '')} | {policy} | {_fmt_num(r.get('trade_count'), 0)} | {_fmt_num(r.get('profit_factor_R'))} | {_fmt_pct(r.get('risk_of_ruin_dd_25pct'))} |"
        )
    lines.append("")
    if not components.empty:
        lines.append("## خلاصه component ها در risk plan منتخب")
        lines.append("")
        lines.append("| Component | Symbol | Side | Trades | Risk weight | Net R weighted | PF_R weighted |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for _, r in components.iterrows():
            lines.append(
                f"| `{r.get('component_id')}` | {r.get('symbol', '')} | {r.get('side', '')} | {_fmt_num(r.get('trade_count'), 0)} | {_fmt_num(r.get('risk_multiplier_mean'))} | {_fmt_num(r.get('net_R_weighted'))}R | {_fmt_num(r.get('profit_factor_R_weighted'))} |"
            )
        lines.append("")
    lines.append("## قاعده اجرایی پیشنهادی")
    lines.append("")
    lines.append("- حالت پیشنهادی برای شروع forward/demo: ریسک پایه ۰.۷۵٪ تا ۱.۰٪.")
    lines.append("- حالت ۱.۵٪ برای گزارش تحقیقاتی یا حالت aggressive قابل بررسی است، نه شروع مستقیم حساب واقعی.")
    lines.append("- معاملات XAUUSD short طبق risk plan منتخب با نصف ریسک پایه محاسبه می شوند.")
    lines.append("- MT5 فقط منبع داده و اجرای سفارش است؛ تصمیم ورود/عدم ورود باید از Python بیاید.")
    lines.append("")
    return "\n".join(lines)


def write_report_bundle(config: Mapping[str, Any]) -> dict[str, Path]:
    fcfg = config.get("final_strategy_report", {})
    if not bool(fcfg.get("enabled", True)):
        raise ValueError("final_strategy_report is disabled in config")
    output_dir = Path(fcfg.get("output_dir", "data/final_strategy_report"))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_csv(fcfg.get("ror_summary_file", "data/inventory_ror_optimization/ror_reduction_summary.csv"), required=True)
    selected = select_strategy_row(summary, fcfg).to_dict()

    core_setup_ids = [str(x) for x in config.get("setup_inventory_portfolio", {}).get("core_setup_ids", [])]
    setups = load_selected_setups(fcfg.get("setup_best_policy_file", "data/setup_inventory/setup_best_viable_policy_by_setup.csv"), core_setup_ids)
    components = selected_component_stress(fcfg.get("component_stress_file", "data/inventory_ror_optimization/ror_component_stress_summary.csv"), selected)
    comparison = select_comparison_rows(summary, selected, fcfg)
    spec = build_live_execution_spec(config=config, selected=selected, setups=setups)
    markdown = render_markdown(selected=selected, comparison=comparison, setups=setups, components=components, spec=spec)

    selected_df = pd.DataFrame([selected])
    selected_df.to_csv(output_dir / "final_strategy_summary.csv", index=False)
    comparison.to_csv(output_dir / "risk_mode_comparison.csv", index=False)
    setups.to_csv(output_dir / "setup_selection_matrix.csv", index=False)
    components.to_csv(output_dir / "component_stress_selected.csv", index=False)
    (output_dir / "final_strategy_report_FA.md").write_text(markdown, encoding="utf-8")
    (output_dir / "live_execution_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "summary": output_dir / "final_strategy_summary.csv",
        "risk_modes": output_dir / "risk_mode_comparison.csv",
        "setup_matrix": output_dir / "setup_selection_matrix.csv",
        "component_stress": output_dir / "component_stress_selected.csv",
        "report": output_dir / "final_strategy_report_FA.md",
        "live_spec": output_dir / "live_execution_spec.json",
    }
