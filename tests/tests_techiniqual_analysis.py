import pandas as pd
import numpy as np
import pytest
# Assuming src.technical_analysis exists and contains compute_indicators.
# For demonstration and direct execution, I'm including the function here.
# In your actual project, ensure 'compute_indicators' is correctly imported.

# --- Corrected compute_indicators (for robust RSI) ---
def compute_indicators(df):
    """
    Calculates various technical indicators for a given DataFrame with 'Close' prices.
    """
    # SMA and EMA
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    # Handle division by zero: replace 0 in avg_loss with NaN
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df

# --- Corrected Pytest Suite ---

@pytest.fixture
def sample_data():
    """
    Provides a DataFrame with enough data points (100 rows)
    for all technical indicators to be calculated and become non-NaN.
    """
    np.random.seed(42)  # For reproducibility
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D'))
    # Simulate realistic-ish price movement
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5 + 0.1)
    return pd.DataFrame({
        'Date': dates,
        'Close': close_prices,
        'Volume': np.random.randint(100000, 1000000, 100) # Dummy volume
    }).set_index('Date') # Indexing by Date is common for time series

def test_sma_calculation(sample_data):
    """
    Tests the Simple Moving Average (SMA) calculation.
    Ensures non-NaN values exist where expected and compares against pandas' own rolling mean.
    """
    df = compute_indicators(sample_data.copy())

    # Ensure SMA_20 produces non-NaN values after the lookback window
    # The first 19 values will be NaN, the 20th and onwards should be numerical
    assert not df['SMA_20'].iloc[19:].isnull().any(), "SMA_20 should have non-NaN values after 19 periods."

    # Compare the last calculated SMA_20 value with pandas' direct calculation
    expected_sma_20 = sample_data['Close'].rolling(window=20).mean().iloc[-1]
    assert np.isclose(df['SMA_20'].iloc[-1], expected_sma_20), "SMA_20 calculation is incorrect."

def test_rsi_range(sample_data):
    """
    Tests that Relative Strength Index (RSI) values are within the valid range [0, 100].
    Only checks non-NaN values.
    """
    df = compute_indicators(sample_data.copy())
    # Drop NaN values from RSI for the check, as initial values will be NaN
    non_nan_rsi = df['RSI'].dropna()

    assert not non_nan_rsi.empty, "RSI column is entirely NaN, indicating calculation issue or insufficient data."
    assert (non_nan_rsi >= 0).all() and (non_nan_rsi <= 100).all(), "RSI values are out of expected range [0, 100]."

def test_macd_columns_and_values(sample_data):
    """
    Tests for the existence of MACD and Signal columns and that they contain non-NaN values.
    """
    df = compute_indicators(sample_data.copy())

    # Check if columns exist
    assert {'MACD', 'Signal'}.issubset(df.columns), "MACD or Signal columns are missing."

    # MACD requires EMA_26 (26 periods) to have a value, so first 25 will be NaN.
    # Signal requires EMA_9 on MACD, so it needs 26 (for MACD) + 9 - 1 = 34 periods.
    assert not df['MACD'].iloc[25:].isnull().any(), "MACD should have non-NaN values after 25 periods."
    assert not df['Signal'].iloc[34:].isnull().any(), "Signal should have non-NaN values after 34 periods."

def test_ema_differs_from_sma(sample_data):
    """
    Tests that EMA_20 and SMA_20 are calculated distinctly for the same window.
    They should generally produce different values due to their different calculation methods.
    """
    df = compute_indicators(sample_data.copy())

    # Select the portion of data where both EMA_20 and SMA_20 have non-NaN values
    # SMA_20 gets first non-NaN at index 19. EMA_20 also gets first non-NaN at index 19.
    comparable_data = df.iloc[19:].dropna(subset=['SMA_20', 'EMA_20'])

    assert not comparable_data.empty, "Insufficient data for meaningful EMA vs SMA comparison."

    # Assert that the EMA_20 and SMA_20 series are not identical across their comparable values.
    # np.allclose checks if two arrays are element-wise equal within a tolerance.
    # We expect them to be *not* all close, meaning they differ.
    assert not np.allclose(comparable_data['EMA_20'], comparable_data['SMA_20'], equal_nan=False), \
           "EMA_20 and SMA_20 should produce different values for the same data."