import numpy as np
import pandas as pd

from src.cointegration import find_cointegrated_pairs, hedge_ratio, spread_half_life


def _make_cointegrated_pair(n=500, beta=1.5, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    common_trend = np.cumsum(rng.normal(0.05, 1.0, n)) + 100
    mean_reverting_noise = np.zeros(n)
    for i in range(1, n):
        mean_reverting_noise[i] = 0.85 * mean_reverting_noise[i - 1] + rng.normal(0, 0.5)

    price_b = pd.Series(common_trend, index=dates)
    price_a = pd.Series(beta * common_trend + mean_reverting_noise, index=dates)
    return price_a, price_b


def _make_independent_walks(n=500, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    walk_a = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100, index=dates)
    walk_b = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100, index=dates)
    return walk_a, walk_b


def test_hedge_ratio_recovers_true_beta():
    price_a, price_b = _make_cointegrated_pair(beta=1.5)
    beta = hedge_ratio(price_a, price_b)
    assert 1.3 < beta < 1.7


def test_cointegrated_pair_is_detected():
    price_a, price_b = _make_cointegrated_pair()
    prices = pd.DataFrame({"A": price_a, "B": price_b})

    pairs = find_cointegrated_pairs(prices, significance=0.05)

    assert len(pairs) == 1
    assert {pairs[0].asset_a, pairs[0].asset_b} == {"A", "B"}
    assert pairs[0].p_value < 0.05


def test_independent_walks_are_not_flagged():
    walk_a, walk_b = _make_independent_walks()
    prices = pd.DataFrame({"A": walk_a, "B": walk_b})

    pairs = find_cointegrated_pairs(prices, significance=0.05)

    assert len(pairs) == 0


def test_half_life_is_finite_for_mean_reverting_spread():
    price_a, price_b = _make_cointegrated_pair(beta=1.5)
    beta = hedge_ratio(price_a, price_b)
    spread = price_a - beta * price_b

    half_life = spread_half_life(spread)

    assert 0 < half_life < 100
