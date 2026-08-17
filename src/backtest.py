"""Backtest engine: turns a position series into P&L with realistic transaction costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    positions: pd.Series
    turnover: float
    n_trades: int


def run_backtest(
    price_a: pd.Series,
    price_b: pd.Series,
    beta: float,
    positions: pd.Series,
    transaction_cost_bps: float = 5.0,
    capital: float = 100_000.0,
) -> BacktestResult:
    """Simulate trading the spread (long A / short beta*B when position=+1).

    A position of +1 means: long 1 unit of A, short `beta` units of B.
    Transaction costs (in basis points) are charged on the *change* in
    notional exposure whenever the position flips, not on every bar, since
    a real broker doesn't charge you for holding a static position.
    """
    ret_a = price_a.pct_change().fillna(0)
    ret_b = price_b.pct_change().fillna(0)

    # dollar-neutral-ish spread return: long A, short beta*B
    spread_return = ret_a - beta * ret_b

    prev_position = positions.shift(1).fillna(0)
    strategy_return = prev_position * spread_return

    position_change = positions.diff().abs().fillna(positions.abs())
    notional_traded = position_change * (1 + abs(beta))
    cost = notional_traded * (transaction_cost_bps / 10_000)

    net_return = strategy_return - cost

    equity_curve = capital * (1 + net_return).cumprod()
    n_trades = int((positions.diff().fillna(0) != 0).sum())
    turnover = float(notional_traded.sum())

    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=net_return,
        positions=positions,
        turnover=turnover,
        n_trades=n_trades,
    )
