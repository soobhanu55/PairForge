"""Cointegration testing and hedge-ratio estimation for candidate pairs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


@dataclass
class PairResult:
    asset_a: str
    asset_b: str
    p_value: float
    hedge_ratio: float
    half_life: float


def hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """OLS hedge ratio: price_a = beta * price_b + const. Returns beta."""
    model = sm.OLS(price_a.values, sm.add_constant(price_b.values)).fit()
    return float(model.params[1])


def spread_half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion via an AR(1) fit on the spread's changes.

    A shorter half-life means the spread reverts to its mean faster, which
    is what makes a pair actually tradeable rather than just statistically
    cointegrated over some long-run sample.
    """
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    lagged, delta = lagged.align(delta, join="inner")

    model = sm.OLS(delta.values, sm.add_constant(lagged.values)).fit()
    theta = model.params[1]
    if theta >= 0:
        return np.inf
    return float(-np.log(2) / theta)


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    significance: float = 0.05,
) -> list[PairResult]:
    """Scan all pairs in `prices` for cointegration via the Engle-Granger test.

    Returns pairs with p-value below `significance`, sorted by p-value
    (strongest cointegration first). Each result includes the OLS hedge
    ratio and the spread's mean-reversion half-life so weak or slow-reverting
    pairs can be filtered out downstream, not just statistically "significant"
    ones.
    """
    results: list[PairResult] = []
    tickers = list(prices.columns)

    for a, b in combinations(tickers, 2):
        series_a, series_b = prices[a], prices[b]
        _, p_value, _ = coint(series_a, series_b)

        if p_value < significance:
            beta = hedge_ratio(series_a, series_b)
            spread = series_a - beta * series_b
            half_life = spread_half_life(spread)
            results.append(PairResult(a, b, p_value, beta, half_life))

    return sorted(results, key=lambda r: r.p_value)


if __name__ == "__main__":
    from data_loader import load_prices

    universe = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE"]
    prices = load_prices(universe, "2019-01-01", "2026-01-01")

    pairs = find_cointegrated_pairs(prices)
    for p in pairs[:10]:
        print(f"{p.asset_a}-{p.asset_b}: p={p.p_value:.4f}, beta={p.hedge_ratio:.3f}, half_life={p.half_life:.1f}d")
