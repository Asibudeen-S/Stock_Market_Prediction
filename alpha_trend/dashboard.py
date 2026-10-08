"""Streamlit dashboard composition."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import streamlit as st

from alpha_trend.charts import candlestick_figure, evaluation_figure
from alpha_trend.config import MODEL_CONFIG
from alpha_trend.data import DataUnavailableError, download_ohlcv, normalize_ticker
from alpha_trend.features import build_features
from alpha_trend.modeling import ModelEvaluation, ModelingError, evaluate_and_forecast

LOGGER = logging.getLogger("alphatrend")


def _format_price(value: float) -> str:
    return f"{value:,.2f}"


def _render_metrics(label: str, metrics, help_text: str) -> None:
    st.markdown(f"#### {label}")
    columns = st.columns(4)
    columns[0].metric("RMSE", _format_price(metrics.rmse), help="Lower is better.")
    columns[1].metric("MAE", _format_price(metrics.mae), help="Lower is better.")
    columns[2].metric("R²", f"{metrics.r2:.3f}", help="Can be negative on unseen data.")
    columns[3].metric(
        "Directional accuracy",
        f"{metrics.directional_accuracy:.1%}",
        help="Share of sessions with the correct Up/Down class.",
    )
    st.caption(help_text)


def _render_forecast(
    evaluation: ModelEvaluation,
    probability_threshold: float,
    minimum_move: float,
) -> None:
    forecast = evaluation.forecast
    st.markdown("### Next-session forecast")
    st.caption(
        "Latest available bar: "
        + forecast.latest_data_date.strftime("%Y-%m-%d")
        + " · Forecast is for the next trading session, not the next calendar day."
    )

    price_column, move_column, confidence_column, direction_column = st.columns(4)
    price_column.metric("Predicted adjusted close", _format_price(forecast.predicted_close))
    move_column.metric(
        "Modeled price change",
        f"{forecast.expected_return:+.2%}",
        help="Regression forecast relative to the latest adjusted close.",
    )
    confidence_column.metric(
        "Model confidence",
        f"{forecast.confidence:.1%}",
        help=(
            "Maximum class probability from XGBoost. This probability is not calibrated "
            "and is not a probability of investment success."
        ),
    )
    direction_column.metric(
        "Direction class",
        forecast.direction,
        help=f"Model-estimated probability of an Up session: {forecast.probability_up:.1%}.",
    )

    signal_text = (
        f"Model signal: {forecast.signal} · Thresholds require at least "
        f"{probability_threshold:.0%} class probability and "
        f"{minimum_move:.2%} modeled move."
    )
    if forecast.signal == "BUY":
        st.success(signal_text)
    elif forecast.signal == "SELL":
        st.error(signal_text)
    else:
        st.info(signal_text)
    st.caption(
        "Buy/Sell/Hold is a mechanical model label for demonstration, not a personalized "
        "recommendation. Forecasts can be wrong."
    )


def _render_evaluation(evaluation: ModelEvaluation, ticker: str) -> None:
    st.markdown("### Model evaluation")
    st.caption(
        "The final chronological holdout is not used to fit its scaler or models. "
        "Walk-forward metrics are calculated on expanding time-series folds inside the "
        "pre-holdout training period."
    )
    _render_metrics(
        "Walk-forward cross-validation",
        evaluation.walk_forward,
        f"{evaluation.cv_folds} expanding folds · scaler refit on each fold's training rows.",
    )
    _render_metrics(
        "Final holdout",
        evaluation.holdout,
        f"{evaluation.training_rows:,} training rows · {evaluation.test_rows:,} holdout rows.",
    )
    st.plotly_chart(
        evaluation_figure(evaluation.test_predictions, ticker),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
        key="holdout-actual-vs-predicted",
    )
    st.caption(
        "Each plotted point is a next-session target date. The holdout model is fixed at "
        "the training boundary; it is not retrained during the holdout."
    )


def _render_sidebar() -> tuple[str, date, date, float, float, bool]:
    today = date.today()
    with st.sidebar:
        st.markdown("## AlphaTrend")
        st.caption("Stock forecasting engine · Daily bars")
        with st.form("forecast-controls"):
            ticker = st.text_input("Ticker", value="AAPL", max_chars=32)
            start_date = st.date_input(
                "Start date",
                value=today - timedelta(days=365 * 5),
                min_value=date(1970, 1, 1),
                max_value=today,
            )
            end_date = st.date_input(
                "End date",
                value=today,
                min_value=date(1970, 1, 1),
                max_value=today,
            )
            probability_threshold = st.slider(
                "Signal probability threshold",
                min_value=0.50,
                max_value=0.80,
                value=MODEL_CONFIG.default_signal_probability,
                step=0.01,
                help="A stricter threshold creates fewer Buy/Sell model labels.",
            )
            minimum_move_percent = st.slider(
                "Minimum modeled move for a signal",
                min_value=0.0,
                max_value=2.0,
                value=MODEL_CONFIG.default_minimum_move * 100,
                step=0.1,
                format="%.1f%%",
            )
            submitted = st.form_submit_button(
                "Run forecast",
                type="primary",
                use_container_width=True,
            )

        st.divider()
        st.markdown("#### Model setup")
        st.markdown(
            "- XGBoost direction classifier\n"
            "- XGBoost next-close regressor\n"
            "- 5 expanding walk-forward folds\n"
            "- Last 20% held out chronologically"
        )
        st.caption("Daily adjusted OHLCV · Data cached for 1 hour")
        st.caption("Signal thresholds affect only the latest forecast label.")

    return (
        ticker,
        start_date,
        end_date,
        probability_threshold,
        minimum_move_percent / 100,
        submitted,
    )


def run_dashboard() -> None:
    st.set_page_config(
        page_title="AlphaTrend · Stock Forecasting Engine",
        page_icon="↗",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
        [data-testid="stMetric"] { background: #fff; border: 1px solid #e9edf4;
          padding: 15px 16px; border-radius: 12px; }
        [data-testid="stMetricLabel"] { color: #667085; }
        div[data-testid="stSidebar"] { border-right: 1px solid #e8ebf2; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    ticker, start_date, end_date, probability_threshold, minimum_move, submitted = (
        _render_sidebar()
    )

    if submitted:
        st.session_state["analysis_request"] = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "probability_threshold": probability_threshold,
            "minimum_move": minimum_move,
        }
    request = st.session_state.get("analysis_request")

    st.title("AlphaTrend")
    st.subheader("Stock Market Forecasting Engine")
    st.markdown(
        "Explore daily price history, technical indicators, and a leakage-aware "
        "next-session model evaluation."
    )
    st.warning(
        "Research and education only. This application is not investment advice. "
        "Forecasts and model signals are uncertain and can result in losses. "
        "Historical backtests do not guarantee future performance."
    )

    if request is None:
        st.info(
            "Choose a ticker and date range in the sidebar, then select **Run forecast**. "
            "A longer history gives the indicators and time-series folds more observations."
        )
        st.markdown(
            "Example symbols: AAPL, NVDA, RELIANCE.NS. "
            "Prices are adjusted by the data provider."
        )
        return

    try:
        symbol = normalize_ticker(request["ticker"])
        start_date = request["start_date"]
        end_date = request["end_date"]
        probability_threshold = request["probability_threshold"]
        minimum_move = request["minimum_move"]
        if start_date >= end_date:
            raise ValueError("The start date must be earlier than the end date.")

        request_key = (
            symbol,
            start_date,
            end_date,
            probability_threshold,
            minimum_move,
        )
        cached_key = st.session_state.get("completed_request_key")
        cached_result = st.session_state.get("completed_analysis")
        if cached_key == request_key and cached_result is not None:
            ohlcv, evaluation, features = cached_result
        else:
            with st.spinner("Downloading adjusted daily prices…"):
                ohlcv = download_ohlcv(symbol, start_date, end_date)
            with st.spinner("Building features and running walk-forward evaluation…"):
                evaluation = evaluate_and_forecast(
                    ohlcv,
                    probability_threshold=probability_threshold,
                    minimum_move=minimum_move,
                )
            features = build_features(ohlcv)
            st.session_state["completed_request_key"] = request_key
            st.session_state["completed_analysis"] = (ohlcv, evaluation, features)
    except (ValueError, DataUnavailableError, ModelingError) as exc:
        st.error(str(exc))
        return
    except Exception:
        LOGGER.exception("Unexpected AlphaTrend analysis error for %s", ticker)
        st.error(
            "The analysis could not be completed. Check the app logs and retry with another "
            "ticker or date range."
        )
        return

    first_date = ohlcv.index[0].strftime("%Y-%m-%d")
    last_date = ohlcv.index[-1].strftime("%Y-%m-%d")
    st.caption(
        f"{symbol} · {first_date} to {last_date} · {len(ohlcv):,} daily bars · "
        "Adjusted OHLCV"
    )

    _render_forecast(evaluation, probability_threshold, minimum_move)
    st.divider()
    st.markdown("### Price history and indicators")
    st.plotly_chart(
        candlestick_figure(ohlcv, features, symbol),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
        key="historical-candlestick",
    )
    st.caption(
        "RSI, MACD, SMAs, Bollinger Bands, and 20-session daily volatility are computed "
        "from current and prior bars only."
    )

    _render_evaluation(evaluation, symbol)
    with st.expander("View latest source rows"):
        st.dataframe(
            ohlcv.join(features[["rsi_14", "macd", "sma_20", "sma_50", "daily_volatility_20"]])
            .tail(20)
            .sort_index(ascending=False),
            width="stretch",
        )

    st.caption(
        "Data source: Yahoo Finance through yfinance. Review the provider's terms and "
        "data rights before redistribution or commercial use."
    )
