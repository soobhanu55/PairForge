import pandas as pd
import pytest

from src.backtest import run_backtest


def _clean_scenario():
    """Round-number prices chosen so every intermediate return is exact,
    making the expected P&L computable by hand rather than approximated.
    """
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    price_a = pd.Series([100.0, 110.0, 121.0, 108.9], index=idx)  # +10%, +10%, -10%
    price_b = pd.Series([50.0, 50.0, 50.0, 50.0], index=idx)  # flat
    positions = pd.Series([0, 1, 1, 0], index=idx)  # enter day1, hold day2, exit day3
    return price_a, price_b, positions


def test_backtest_matches_hand_calculated_equity_curve():
    price_a, price_b, positions = _clean_scenario()

    result = run_backtest(
        price_a, price_b, beta=1.0, positions=positions,
        transaction_cost_bps=5.0, capital=100_000.0,
    )

    expected_equity = [100_000, 99_900, 109_890, 98_791.11]
    for actual, expected in zip(result.equity_curve, expected_equity):
        assert actual == pytest.approx(expected, rel=1e-6)


def test_backtest_counts_trades_and_turnover_correctly():
    price_a, price_b, positions = _clean_scenario()

    result = run_backtest(price_a, price_b, beta=1.0, positions=positions)

    assert result.n_trades == 2  # one entry, one exit
    assert result.turnover == pytest.approx(4.0)  # (1+|beta|) notional per flip, twice


def test_zero_transaction_cost_matches_raw_strategy_return():
    price_a, price_b, positions = _clean_scenario()

    result = run_backtest(
        price_a, price_b, beta=1.0, positions=positions, transaction_cost_bps=0.0,
    )

    # with no cost, the only nonzero returns are the two days actually held
    assert result.daily_returns.iloc[1] == pytest.approx(0.0)
    assert result.daily_returns.iloc[2] == pytest.approx(0.10)
    assert result.daily_returns.iloc[3] == pytest.approx(-0.10)


def test_higher_transaction_cost_reduces_final_equity():
    price_a, price_b, positions = _clean_scenario()

    cheap = run_backtest(price_a, price_b, 1.0, positions, transaction_cost_bps=1.0)
    expensive = run_backtest(price_a, price_b, 1.0, positions, transaction_cost_bps=50.0)

    assert expensive.equity_curve.iloc[-1] < cheap.equity_curve.iloc[-1]


def test_flat_positions_produce_no_trades_and_flat_equity():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    price_a = pd.Series([100.0, 105.0, 95.0, 102.0], index=idx)
    price_b = pd.Series([50.0, 52.0, 48.0, 51.0], index=idx)
    positions = pd.Series([0, 0, 0, 0], index=idx)

    result = run_backtest(price_a, price_b, beta=1.0, positions=positions)

    assert result.n_trades == 0
    assert result.turnover == 0.0
    assert (result.equity_curve == 100_000.0).all()
