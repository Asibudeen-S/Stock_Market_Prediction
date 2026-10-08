"""Past-only technical indicator and feature calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "return_1d",
    "log_return_1d",
    "intraday_range",
    "overnight_gap",
    "volume_change",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "sma_20",
    "sma_50",
    "bollinger_upper",
    "bollinger_lower",
    "bollinger_width",
    "daily_volatility_20",
]


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Calculate causal indicators using only data available on each row's date."""
    if ohlcv.empty:
        raise ValueError("OHLCV data is empty.")

    frame = ohlcv.copy().sort_index()
    close = frame["Close"].astype("float64")
    high = frame["High"].astype("float64")
    low = frame["Low"].astype("float64")
    open_price = frame["Open"].astype("float64")
    volume = frame["Volume"].astype("float64")

    frame["return_1d"] = close.pct_change(fill_method=None)
    frame["log_return_1d"] = np.log(close).diff()
    frame["intraday_range"] = (high - low) / close
    previous_close = close.shift(1)
    frame["overnight_gap"] = (open_price - previous_close) / previous_close
    frame["volume_change"] = np.log1p(volume).diff()

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50.0)
    frame["rsi_14"] = rsi

    ema_12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    frame["macd"] = ema_12 - ema_26
    frame["macd_signal"] = frame["macd"].ewm(
        span=9, min_periods=9, adjust=False
    ).mean()
    frame["macd_histogram"] = frame["macd"] - frame["macd_signal"]

    frame["sma_20"] = close.rolling(window=20, min_periods=20).mean()
    frame["sma_50"] = close.rolling(window=50, min_periods=50).mean()
    rolling_std = close.rolling(window=20, min_periods=20).std(ddof=0)
    frame["bollinger_upper"] = frame["sma_20"] + 2 * rolling_std
    frame["bollinger_lower"] = frame["sma_20"] - 2 * rolling_std
    frame["bollinger_width"] = (
        (frame["bollinger_upper"] - frame["bollinger_lower"])
        / frame["sma_20"].replace(0, np.nan)
    )

    frame["daily_volatility_20"] = frame["return_1d"].rolling(
        window=20, min_periods=20
    ).std(ddof=1)

    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return frame
