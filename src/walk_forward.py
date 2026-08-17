"""Walk-forward validation: re-select the pair and hedge ratio on a rolling
in-sample window, then trade it out-of-sample on the following window.

This is the piece that keeps the strategy honest. Selecting a cointegrated
pair once on the full history and backtesting on that same history is
look-ahead bias: the pair was chosen *because* it worked over that period.
Walk-forward re-runs pair selection on each formation window using only
data available at that point in time, then evaluates performance strictly
on the unseen period that follows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.backtest import run_backtest
from src.cointegration import find_cointegrated_pairs
from src.metrics import summarize
from src.signal import compute_spread, generate_signals, rolling_zscore


@dataclass
class WalkForwardResult:
    window_results: list[dict] = field(default_factory=list)
    out_of_sample_equity: pd.Series | None = None


def walk_forward_validate(
    prices: pd.DataFrame,
    formation_window: int = 252,
    trading_window: int = 63,
    zscore_window: int = 30,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    transaction_cost_bps: float = 5.0,
) -> WalkForwardResult:
    result = WalkForwardResult()
    equity_segments: list[pd.Series] = []
    capital = 100_000.0

    n = len(prices)
    start = 0

    while start + formation_window + trading_window <= n:
        formation = prices.iloc[start : start + formation_window]
        trading = prices.iloc[start + formation_window : start + formation_window + trading_window]

        pairs = find_cointegrated_pairs(formation, significance=0.05)
        if not pairs:
            start += trading_window
            continue

        best = pairs[0]
        a, b, beta = best.asset_a, best.asset_b, best.hedge_ratio

        # zscore needs some formation-period history to warm up its rolling
        # window, so carry the tail of the formation period into the
        # out-of-sample slice purely for that calculation, not for signals.
        warmup = formation[[a, b]].tail(zscore_window)
        oos_prices = pd.concat([warmup, trading[[a, b]]])

        spread = compute_spread(oos_prices[a], oos_prices[b], beta)
        zscore = rolling_zscore(spread, window=zscore_window)
        positions = generate_signals(zscore, entry_threshold, exit_threshold)

        # drop the warmup rows before backtesting so we only trade/report
        # on the genuinely out-of-sample trading window
        trading_positions = positions.loc[trading.index[0] :]
        trading_a = oos_prices[a].loc[trading.index[0] :]
        trading_b = oos_prices[b].loc[trading.index[0] :]

        bt = run_backtest(
            trading_a, trading_b, beta, trading_positions,
            transaction_cost_bps=transaction_cost_bps, capital=capital,
        )

        window_summary = summarize(bt.equity_curve, bt.daily_returns, bt.turnover, bt.n_trades)
        window_summary.update({
            "pair": f"{a}-{b}",
            "window_start": str(trading.index[0].date()),
            "window_end": str(trading.index[-1].date()),
        })
        result.window_results.append(window_summary)

        equity_segments.append(bt.equity_curve)
        capital = float(bt.equity_curve.iloc[-1])
        start += trading_window

    if equity_segments:
        result.out_of_sample_equity = pd.concat(equity_segments)

    return result


if __name__ == "__main__":
    from src.data_loader import load_prices

    universe = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE"]
    prices = load_prices(universe, "2019-01-01", "2026-01-01")

    wf = walk_forward_validate(prices)
    for w in wf.window_results:
        print(w)

    if wf.out_of_sample_equity is not None:
        overall = summarize(
            wf.out_of_sample_equity,
            wf.out_of_sample_equity.pct_change().fillna(0),
            turnover=sum(w["turnover"] for w in wf.window_results),
            n_trades=sum(w["n_trades"] for w in wf.window_results),
        )
        print("\nOverall out-of-sample performance:", overall)
