from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debco.trading.inventory_ror_optimizer import (
    DEFAULT_RISK_PER_TRADE_GRID,
    DecisionFilters,
    apply_decision_filters,
    component_risk_summary,
    discover_selected_trade_files,
    load_source_summary,
    summarize_weighted_trades,
)
from debco.utils.io import ensure_dir, read_json


def _rel_to_portfolio_policy(path: Path, root: Path) -> tuple[str, str]:
    rel = path.relative_to(root).parts
    if len(rel) < 3:
        return "unknown_portfolio", "unknown_policy"
    return str(rel[0]), str(rel[1])


def _summary_lookup(summary_df: pd.DataFrame) -> dict[tuple[str, str], dict]:
    if summary_df.empty or not {"portfolio", "risk_policy"}.issubset(summary_df.columns):
        return {}
    return {
        (str(row["portfolio"]), str(row["risk_policy"])): row.to_dict()
        for _, row in summary_df.iterrows()
    }


def _parse_float_list(values: list[str] | None) -> list[float] | None:
    if not values:
        return None
    return [float(x) for x in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize inventory-portfolio risk-of-ruin by sweeping risk per trade and exposure weights.")
    parser.add_argument("--config", required=True, help="Path to ml config JSON.")
    parser.add_argument("--source-dir", default=None, help="Override inventory portfolio eval directory.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--risk-per-trades", nargs="*", default=None, help="Optional risk-per-trade grid, e.g. 0.005 0.0075 0.01.")
    parser.add_argument("--portfolio", default=None, help="Optional portfolio name filter.")
    parser.add_argument("--risk-policy", default=None, help="Optional risk policy filter.")
    args = parser.parse_args()

    config = read_json(args.config)
    roc = config.get("inventory_ror_optimizer", {})
    if not bool(roc.get("enabled", True)):
        raise SystemExit("inventory_ror_optimizer is disabled in config.")

    source_dir = Path(args.source_dir or roc.get("source_dir", "data/inventory_portfolio_eval"))
    output_dir = ensure_dir(Path(args.output_dir or roc.get("output_dir", "data/inventory_ror_optimization")))
    risk_grid = _parse_float_list(args.risk_per_trades) or [float(x) for x in roc.get("risk_per_trade_grid", DEFAULT_RISK_PER_TRADE_GRID)]
    risk_plans = list(roc.get("risk_plans", [])) or [{"name": "uniform", "description": "Same risk for all trades."}]
    ruin_drawdowns = [float(x) for x in roc.get("ruin_drawdowns", [0.25, 0.30])]
    simulations = int(roc.get("risk_of_ruin_simulations", 5000))
    seed = int(roc.get("random_seed", 42))
    initial_capital = float(roc.get("initial_capital", config.get("portfolio_eval", {}).get("initial_capital", 1000.0)))

    filters_cfg = roc.get("decision_filters", {})
    filters = DecisionFilters(
        min_trades=float(filters_cfg.get("min_trades", 60)),
        min_profit_factor_R=float(filters_cfg.get("min_profit_factor_R", 1.50)),
        max_risk_of_ruin_dd_25pct=float(filters_cfg.get("max_risk_of_ruin_dd_25pct", 0.01)),
        min_positive_folds=float(filters_cfg.get("min_positive_folds", 6)),
        require_all_components_loaded=bool(filters_cfg.get("require_all_components_loaded", True)),
        require_side_complete_3x3=bool(filters_cfg.get("require_side_complete_3x3", False)),
    )

    files = discover_selected_trade_files(source_dir)
    if args.portfolio:
        files = [p for p in files if _rel_to_portfolio_policy(p, source_dir)[0] == args.portfolio]
    if args.risk_policy:
        files = [p for p in files if _rel_to_portfolio_policy(p, source_dir)[1] == args.risk_policy]
    if not files:
        raise SystemExit(f"No portfolio_selected_trades.csv files found under {source_dir}.")

    source_summary = load_source_summary(source_dir)
    summary_map = _summary_lookup(source_summary)

    rows = []
    component_rows = []
    for file_path in files:
        portfolio, risk_policy = _rel_to_portfolio_policy(file_path, source_dir)
        trades = pd.read_csv(file_path)
        source_row = summary_map.get((portfolio, risk_policy), {})
        for plan in risk_plans:
            plan_name = str(plan.get("name", "risk_plan"))
            comp = component_risk_summary(trades, plan)
            if not comp.empty:
                comp.insert(0, "portfolio", portfolio)
                comp.insert(1, "risk_policy", risk_policy)
                component_rows.append(comp)
            for risk in risk_grid:
                rows.append(
                    summarize_weighted_trades(
                        trades,
                        portfolio=portfolio,
                        risk_policy=risk_policy,
                        risk_plan=plan,
                        risk_per_trade=float(risk),
                        initial_capital=initial_capital,
                        ruin_drawdowns=ruin_drawdowns,
                        simulations=simulations,
                        seed=seed,
                        source_summary=source_row,
                    )
                )
            print(f"optimized {portfolio} | {risk_policy} | plan={plan_name}")

    sweep = pd.DataFrame(rows)
    sweep = apply_decision_filters(sweep, filters)
    sort_cols = [c for c in ["portfolio_pass", "side_complete_3x3_configured", "risk_of_ruin_dd_25pct", "profit_factor_R", "net_return_pct_on_initial"] if c in sweep.columns]
    ascending = [False, False, True, False, False][: len(sort_cols)]
    if sort_cols:
        sweep = sweep.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    sweep.to_csv(output_dir / "ror_reduction_summary.csv", index=False)

    passed = sweep.loc[sweep["portfolio_pass"]].copy()
    passed.to_csv(output_dir / "ror_reduction_pass.csv", index=False)

    # One recommended row per portfolio/risk-policy: highest risk that still passes; tie-break by return.
    if not passed.empty:
        rec = passed.sort_values(
            ["portfolio", "risk_policy", "risk_per_trade_pct", "net_return_pct_on_initial"],
            ascending=[True, True, False, False],
        ).groupby(["portfolio", "risk_policy"], as_index=False).head(1)
    else:
        rec = pd.DataFrame()
    rec.to_csv(output_dir / "recommended_risk_plan.csv", index=False)

    components = pd.concat(component_rows, ignore_index=True) if component_rows else pd.DataFrame()
    components.to_csv(output_dir / "ror_component_stress_summary.csv", index=False)

    focus = [
        c
        for c in [
            "portfolio_pass",
            "portfolio",
            "risk_policy",
            "risk_plan",
            "risk_per_trade_pct",
            "trade_count",
            "win_rate",
            "profit_factor_R",
            "net_return_pct_on_initial",
            "max_drawdown_pct",
            "risk_of_ruin_dd_25pct",
            "positive_folds",
            "loaded_component_count",
            "configured_component_count",
            "side_complete_3x3_configured",
            "avg_risk_multiplier",
        ]
        if c in sweep.columns
    ]
    print("\n--- RISK OF RUIN REDUCTION SUMMARY ---")
    print(sweep[focus].head(40).to_string(index=False))
    if rec.empty:
        print("\nNo risk plan passed the requested filters.")
    else:
        print("\n--- RECOMMENDED RISK PLANS ---")
        print(rec[focus].to_string(index=False))
    print(f"\nsaved: {output_dir / 'ror_reduction_summary.csv'}")


if __name__ == "__main__":
    main()
