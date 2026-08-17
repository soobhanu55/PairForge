"""Fetch and cache historical adjusted-close price data for a universe of tickers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices, columns=tickers, index=date.

    Caches to data/prices_<start>_<end>.csv so repeated runs don't re-hit the
    network. Tickers with more than 5% missing data over the window are
    dropped rather than silently forward-filled across large gaps.
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache_path = DATA_DIR / f"prices_{start}_{end}.csv"

    if cache and cache_path.exists():
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        missing_tickers = [t for t in tickers if t not in prices.columns]
        if not missing_tickers:
            return prices[tickers]

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)})
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.95))
    prices = prices.ffill().dropna()

    if cache:
        prices.to_csv(cache_path)

    return prices


if __name__ == "__main__":
    universe = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE"]
    df = load_prices(universe, "2019-01-01", "2026-01-01")
    print(df.shape)
    print(df.head())
