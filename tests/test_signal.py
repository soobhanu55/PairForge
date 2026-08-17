import pandas as pd

from src.signal import generate_signals, rolling_zscore


def test_rolling_zscore_flat_series_is_nan_then_zero():
    spread = pd.Series([10.0] * 40)
    z = rolling_zscore(spread, window=10)

    assert z.iloc[:9].isna().all()
    # zero variance in the window -> division by zero -> NaN, not an exception
    assert pd.isna(z.iloc[15])


def test_generate_signals_enters_long_on_low_zscore():
    z = pd.Series([0.0, 0.0, -2.5, -2.5, -2.5, 0.0, 0.0])
    positions = generate_signals(z, entry_threshold=2.0, exit_threshold=0.5)

    assert positions.iloc[2] == 1  # enters long-spread once z crosses -2.0
    assert positions.iloc[4] == 1  # stays in position while z remains below -0.5
    assert positions.iloc[6] == 0  # exits once z reverts above -0.5


def test_generate_signals_enters_short_on_high_zscore():
    z = pd.Series([0.0, 0.0, 2.5, 2.5, 0.4, 0.0])
    positions = generate_signals(z, entry_threshold=2.0, exit_threshold=0.5)

    assert positions.iloc[2] == -1
    assert positions.iloc[4] == 0  # exits once z drops below +0.5


def test_generate_signals_no_entry_within_threshold():
    z = pd.Series([0.1, -0.3, 0.5, -0.5, 1.0])
    positions = generate_signals(z, entry_threshold=2.0, exit_threshold=0.5)

    assert (positions == 0).all()
