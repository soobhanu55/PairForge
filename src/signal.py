"""Spread z-score construction and entry/exit signal generation."""

from __future__ import annotations

import pandas as pd


def compute_spread(price_a: pd.Series, price_b: pd.Series, beta: float) -> pd.Series:
    return price_a - beta * price_b


def rolling_zscore(spread: pd.Series, window: int = 30) -> pd.Series:
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std


def generate_signals(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> pd.Series:
    """Return a position series in {-1, 0, +1} for the spread (long A/short B = +1).

    Enter long-spread when zscore < -entry_threshold (spread unusually low,
    expect it to rise). Enter short-spread when zscore > entry_threshold.
    Exit once the zscore reverts inside +/- exit_threshold. Positions persist
    between entry and exit rather than being re-evaluated bar by bar, which
    avoids flickering in and out of a trade on noise near the threshold.
    """
    position = pd.Series(0, index=zscore.index, dtype=int)
    current = 0

    for i, z in enumerate(zscore):
        if pd.isna(z):
            position.iloc[i] = current
            continue

        if current == 0:
            if z < -entry_threshold:
                current = 1
            elif z > entry_threshold:
                current = -1
        elif current == 1 and z > -exit_threshold:
            current = 0
        elif current == -1 and z < exit_threshold:
            current = 0

        position.iloc[i] = current

    return position
