"""Performance metrics: Sharpe ratio, max drawdown, CAGR, turnover."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / excess.std())


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min())


def cagr(equity_curve: pd.Series) -> float:
    n_years = len(equity_curve) / TRADING_DAYS_PER_YEAR
    if n_years <= 0 or equity_curve.iloc[0] <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    return float(total_return ** (1 / n_years) - 1)


def summarize(equity_curve: pd.Series, daily_returns: pd.Series, turnover: float, n_trades: int) -> dict:
    return {
        "sharpe_ratio": round(sharpe_ratio(daily_returns), 3),
        "max_drawdown": round(max_drawdown(equity_curve), 4),
        "cagr": round(cagr(equity_curve), 4),
        "total_return": round(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1, 4),
        "turnover": round(turnover, 2),
        "n_trades": n_trades,
    }
