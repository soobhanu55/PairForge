"""End-to-end pipeline: fetch data -> find cointegrated pairs -> backtest ->
walk-forward validate -> report results and save an equity curve plot.

Run: python main.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def main() -> None:
    print(f"Loading prices for {UNIVERSE} from {START} to {END}...")
    prices = load_prices(UNIVERSE, START, END)
    print(f"Loaded {prices.shape[0]} trading days x {prices.shape[1]} tickers\n")

    print("Scanning for cointegrated pairs (in-sample, full history)...")
    pairs = find_cointegrated_pairs(prices)
    if not pairs:
        print("No cointegrated pairs found at p < 0.05 in this universe/window.")
        return

    print(f"Found {len(pairs)} cointegrated pair(s):")
    for p in pairs[:5]:
        print(f"  {p.asset_a}-{p.asset_b}: p={p.p_value:.4f}, beta={p.hedge_ratio:.3f}, half_life={p.half_life:.1f}d")

    best = pairs[0]
    print(f"\nBest pair by p-value: {best.asset_a}-{best.asset_b}")

    # Naive full-history backtest -- this is optimistic, since the pair was
    # selected using the same data it's then evaluated on. Reported here
    # only as a contrast to the honest walk-forward number below.
    spread = compute_spread(prices[best.asset_a], prices[best.asset_b], best.hedge_ratio)
    zscore = rolling_zscore(spread, window=ZSCORE_WINDOW)
    positions = generate_signals(zscore, ENTRY_THRESHOLD, EXIT_THRESHOLD)
    bt = run_backtest(
        prices[best.asset_a], prices[best.asset_b], best.hedge_ratio, positions,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )
    naive_summary = summarize(bt.equity_curve, bt.daily_returns, bt.turnover, bt.n_trades)
    print("\n[Naive, in-sample-selected] full-history backtest:")
    for k, v in naive_summary.items():
        print(f"  {k}: {v}")

    print("\nRunning walk-forward validation (out-of-sample pair re-selection)...")
    wf = walk_forward_validate(
        prices,
        zscore_window=ZSCORE_WINDOW,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )

    if not wf.window_results:
        print("Walk-forward produced no tradeable windows -- try a longer date range or looser significance.")
        return

    print(f"\n{len(wf.window_results)} walk-forward windows:")
    for w in wf.window_results:
        print(f"  {w['window_start']} to {w['window_end']} | pair={w['pair']} | "
              f"sharpe={w['sharpe_ratio']} | return={w['total_return']} | trades={w['n_trades']}")

    overall = summarize(
        wf.out_of_sample_equity,
        wf.out_of_sample_equity.pct_change().fillna(0),
        turnover=sum(w["turnover"] for w in wf.window_results),
        n_trades=sum(w["n_trades"] for w in wf.window_results),
    )
    print("\n[Honest] Overall walk-forward out-of-sample performance:")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    fig, ax = plt.subplots(figsize=(10, 5))
    bt.equity_curve.plot(ax=ax, label=f"Naive in-sample ({best.asset_a}-{best.asset_b})")
    wf.out_of_sample_equity.plot(ax=ax, label="Walk-forward out-of-sample")
    ax.set_title("Pairs Trading: Naive vs. Walk-Forward Equity Curve")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("equity_curve.png", dpi=150)
    print("\nSaved equity_curve.png")


if __name__ == "__main__":
    main()
