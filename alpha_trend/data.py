"""Market-data retrieval and normalization."""

from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

from alpha_trend.config import DATA_CACHE_TTL_SECONDS, MAX_TICKER_LENGTH

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
TICKER_PATTERN = re.compile(r"^[A-Za-z0-9.^=_-]+$")


class DataUnavailableError(RuntimeError):
    """Raised when a ticker or date range has no usable market data."""


def normalize_ticker(ticker: str) -> str:
    """Validate and normalize a Yahoo Finance ticker symbol."""
    normalized = ticker.strip().upper()
    if (
        not normalized
        or len(normalized) > MAX_TICKER_LENGTH
        or not TICKER_PATTERN.fullmatch(normalized)
    ):
        raise ValueError(
            "Enter a ticker using letters, numbers, dots, dashes, carets, or equals signs."
        )
    return normalized


def _flatten_single_ticker_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance's flat or MultiIndex columns to OHLCV names."""
    if not isinstance(frame.columns, pd.MultiIndex):
        result = frame.copy()
        result.columns = [str(column).title() for column in result.columns]
        return result

    expected = set(REQUIRED_COLUMNS)
    selected_level: int | None = None
    for level in range(frame.columns.nlevels):
        labels = {str(value).title() for value in frame.columns.get_level_values(level)}
        if expected.issubset(labels):
            selected_level = level
            break

    if selected_level is None:
        raise DataUnavailableError("The market-data provider returned an unexpected column layout.")

    result = frame.copy()
    result.columns = [
        str(value).title() for value in result.columns.get_level_values(selected_level)
    ]
    return result.loc[:, ~result.columns.duplicated()]


@st.cache_data(ttl=DATA_CACHE_TTL_SECONDS, max_entries=128, show_spinner=False)
def download_ohlcv(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Download daily adjusted OHLCV data; end_date is inclusive in the UI."""
    symbol = normalize_ticker(ticker)
    if start_date >= end_date:
        raise ValueError("The start date must be earlier than the end date.")

    try:
        # yfinance treats end as exclusive, so add one day to include the selected UI date.
        downloaded = yf.download(
            symbol,
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=False,
            timeout=20,
            multi_level_index=False,
        )
    except Exception as exc:
        raise DataUnavailableError(
            "Could not retrieve market data. Check the ticker, date range, and network, then retry."
        ) from exc

    if downloaded is None or downloaded.empty:
        raise DataUnavailableError(
            f"No daily prices were returned for {symbol}. Check the ticker and date range."
        )

    frame = _flatten_single_ticker_columns(downloaded)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataUnavailableError(
            "The market-data response is missing required fields: " + ", ".join(missing)
        )

    frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    frame.index.name = "Date"
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.dropna(subset=list(REQUIRED_COLUMNS))
    frame = frame[(frame["Close"] > 0) & (frame["Volume"] >= 0)]

    if frame.empty:
        raise DataUnavailableError(f"No valid OHLCV rows were returned for {symbol}.")

    return frame
