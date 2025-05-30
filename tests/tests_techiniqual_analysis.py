import pandas as pd
import numpy as np
import pytest
from src.technical_analysis import compute_indicators

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Close': [10, 12, 11, 13, 14, 15, 16, 17, 18, 19],
        'Volume': [100] * 10  # Dummy volume
    })

def test_sma_calculation(sample_data):
    df = compute_indicators(sample_data.copy())
    assert df['SMA_20'].iloc[-1] == sample_data['Close'].rolling(20).mean().iloc[-1]  # Last SMA value

def test_rsi_range(sample_data):
    df = compute_indicators(sample_data.copy())
    assert (df['RSI'] >= 0).all() and (df['RSI'] <= 100).all()  # RSI bounds

def test_macd_columns(sample_data):
    df = compute_indicators(sample_data.copy())
    assert {'MACD', 'Signal'}.issubset(df.columns)  # MACD columns exist

def test_ema_faster_than_sma(sample_data):
    df = compute_indicators(sample_data.copy())
    assert (df['EMA_20'] != df['SMA_20']).any()  # EMA reacts faster