"""Interactive Plotly charts for prices and out-of-sample model evaluation."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def candlestick_figure(
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """Create an adjusted-price candlestick chart with technical overlays and volume."""
    frame = ohlcv.join(
        features[
            [
                "sma_20",
                "sma_50",
                "bollinger_upper",
                "bollinger_lower",
            ]
        ],
        how="left",
        rsuffix="_feature",
    )

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.76, 0.24],
    )
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["bollinger_upper"],
            name="Bollinger upper",
            line={"color": "rgba(101, 116, 159, 0.34)", "width": 1},
            hovertemplate="Upper band: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["bollinger_lower"],
            name="Bollinger band",
            line={"color": "rgba(101, 116, 159, 0.34)", "width": 1},
            fill="tonexty",
            fillcolor="rgba(101, 116, 159, 0.08)",
            hovertemplate="Lower band: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["Open"],
            high=frame["High"],
            low=frame["Low"],
            close=frame["Close"],
            name="Adjusted OHLC",
            increasing_line_color="#19a982",
            decreasing_line_color="#e36772",
            increasing_fillcolor="#19a982",
            decreasing_fillcolor="#e36772",
            whiskerwidth=0.35,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["sma_20"],
            name="SMA 20",
            line={"color": "#f0a34b", "width": 1.4},
            hovertemplate="SMA 20: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["sma_50"],
            name="SMA 50",
            line={"color": "#5368e8", "width": 1.5},
            hovertemplate="SMA 50: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    volume_colors = [
        "#19a982" if close >= open_price else "#e36772"
        for open_price, close in zip(frame["Open"], frame["Close"], strict=True)
    ]
    figure.add_trace(
        go.Bar(
            x=frame.index,
            y=frame["Volume"],
            name="Volume",
            marker_color=volume_colors,
            opacity=0.62,
            hovertemplate="Volume: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    figure.update_layout(
        title={"text": ticker + " · Adjusted OHLCV", "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        height=620,
        margin={"l": 12, "r": 16, "t": 54, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.015,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 10},
        },
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        font={"family": "Arial, sans-serif", "color": "#344054"},
    )
    figure.update_xaxes(showgrid=False, linecolor="#e5e9f0")
    figure.update_yaxes(title_text="Price", row=1, col=1, gridcolor="#edf0f5")
    figure.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="#edf0f5")
    return figure


def evaluation_figure(predictions: pd.DataFrame, ticker: str) -> go.Figure:
    """Compare actual and predicted next-session prices across the holdout period."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predictions["actual_close"],
            name="Actual",
            mode="lines",
            line={"color": "#26344a", "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Actual: %{y:.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predictions["predicted_close"],
            name="Predicted",
            mode="lines",
            line={"color": "#5368e8", "width": 1.8, "dash": "dot"},
            hovertemplate="%{x|%Y-%m-%d}<br>Predicted: %{y:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": ticker + " · Holdout next-session price", "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        height=390,
        margin={"l": 12, "r": 16, "t": 54, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
        font={"family": "Arial, sans-serif", "color": "#344054"},
    )
    figure.update_xaxes(showgrid=False, linecolor="#e5e9f0", title_text="Target trading session")
    figure.update_yaxes(title_text="Adjusted closing price", gridcolor="#edf0f5")
    return figure
