"""Run the real pipeline once and cache all results to results.json so the
Streamlit dashboard can load instantly instead of re-fetching/re-computing
on every page load. The numbers here are identical to what `python main.py`
produces, just serialized for the UI.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.backtest import run_backtest
from src.cointegration import find_cointegrated_pairs
from src.data_loader import load_prices
from src.metrics import summarize
from src.signal import compute_spread, generate_signals, rolling_zscore
from src.walk_forward import walk_forward_validate

UNIVERSE = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE"]
START, END = "2019-01-01", "2026-01-01"
ZSCORE_WINDOW = 30
ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 0.5
TRANSACTION_COST_BPS = 5.0

OUT_PATH = Path(__file__).resolve().parent.parent / "results.json"


def main() -> None:
    prices = load_prices(UNIVERSE, START, END)

    pairs = find_cointegrated_pairs(prices)
    pairs_serialized = [
        {"asset_a": p.asset_a, "asset_b": p.asset_b, "p_value": round(p.p_value, 4),
         "hedge_ratio": round(p.hedge_ratio, 3), "half_life": round(p.half_life, 1)}
        for p in pairs
    ]

    best = pairs[0]
    spread = compute_spread(prices[best.asset_a], prices[best.asset_b], best.hedge_ratio)
    zscore = rolling_zscore(spread, window=ZSCORE_WINDOW)
    positions = generate_signals(zscore, ENTRY_THRESHOLD, EXIT_THRESHOLD)
    naive_bt = run_backtest(
        prices[best.asset_a], prices[best.asset_b], best.hedge_ratio, positions,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )
    naive_summary = summarize(naive_bt.equity_curve, naive_bt.daily_returns, naive_bt.turnover, naive_bt.n_trades)

    wf = walk_forward_validate(
        prices, zscore_window=ZSCORE_WINDOW, entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD, transaction_cost_bps=TRANSACTION_COST_BPS,
    )
    wf_summary = summarize(
        wf.out_of_sample_equity,
        wf.out_of_sample_equity.pct_change().fillna(0),
        turnover=sum(w["turnover"] for w in wf.window_results),
        n_trades=sum(w["n_trades"] for w in wf.window_results),
    )

    results = {
        "universe": UNIVERSE,
        "date_range": {"start": START, "end": END},
        "pairs": pairs_serialized,
        "best_pair": f"{best.asset_a}-{best.asset_b}",
        "naive": {
            "summary": naive_summary,
            "equity_curve": {
                "dates": [str(d.date()) for d in naive_bt.equity_curve.index],
                "values": [round(v, 2) for v in naive_bt.equity_curve.values],
            },
        },
        "walk_forward": {
            "summary": wf_summary,
            "windows": wf.window_results,
            "equity_curve": {
                "dates": [str(d.date()) for d in wf.out_of_sample_equity.index],
                "values": [round(v, 2) for v in wf.out_of_sample_equity.values],
            },
        },
    }

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
