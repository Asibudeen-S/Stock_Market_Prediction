# AlphaTrend: Stock Market Forecasting Engine

AlphaTrend is a Streamlit research dashboard that downloads daily Yahoo Finance history, calculates past-only technical indicators, evaluates two XGBoost models with walk-forward validation and a chronological holdout, and forecasts the next trading session.

## Run in VS Code

1. Open the repository in VS Code.
2. In the integrated terminal, change into the AlphaTrend folder:

   cd alphatrend

3. Use Python 3.11 or newer to create and activate a virtual environment:

   python3 -m venv .venv

   macOS/Linux: source .venv/bin/activate

   Windows PowerShell: .venv\Scripts\Activate.ps1

4. Install the pinned application dependencies:

   python -m pip install -r requirements.txt

5. On macOS, install XGBoost's system OpenMP runtime once:

   brew install libomp

   Linux Docker installs its OpenMP runtime in the image. On native Linux, install
   the distribution's `libgomp` package.

6. Start Streamlit from the virtual environment:

   macOS/Linux: .venv/bin/python -m streamlit run app.py

   Windows PowerShell: .venv\Scripts\python.exe -m streamlit run app.py

6. Open the local URL printed in the terminal, normally http://localhost:8501.

Docker is an alternative: from this folder run docker compose up --build, then open http://localhost:8501.

### macOS troubleshooting

If XGBoost raises `Library not loaded: @rpath/libomp.dylib`, install OpenMP with
`brew install libomp`, stop the current Streamlit process, then start it again from
the `.venv` interpreter. Reinstalling the Python packages is not normally needed.

## Modeling design

- Daily adjusted OHLCV is downloaded for the user-selected ticker and inclusive date range. The provider's end date is exclusive, so ingestion adds one day.
- Features are strictly trailing: RSI(14), MACD(12, 26, 9), SMA(20), SMA(50), Bollinger Bands(20, 2), 20-session rolling standard deviation of daily returns, and price/volume-derived features.
- The targets are next-row adjusted close and next-session Up/Down direction.
- The final chronological holdout is separated from training by one observation. The one-row gap is required because the label for feature row t is the close at t+1.
- Walk-forward folds use TimeSeriesSplit(gap=1). Every fold fits a fresh MinMaxScaler on that fold's training rows only. The holdout scaler is fit only on pre-holdout training rows. A final scaler and model are fit on all labeled history for the next-session forecast.
- XGBoost is tree based and does not generally require MinMax scaling; the scaler is retained here to satisfy the requested feature-processing workflow. Tree models can still extrapolate poorly, especially on price levels.
- Holdout RMSE, MAE, R², and classifier directional accuracy are reported alongside out-of-fold walk-forward metrics. The evaluation chart shows next-session test targets against predictions from the fixed holdout model.
- The confidence score is the maximum class probability from XGBoost and is not calibrated. Buy/Sell/Hold is a threshold-based model label, not an investment recommendation.

## Data and financial-use limitations

yfinance is a community-maintained interface to Yahoo Finance, not an official Yahoo commercial market-data API. Its [project documentation](https://github.com/ranaroussi/yfinance/blob/main/README.md) says downloaded data is intended for personal use and points users to [Yahoo's terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html). Confirm the applicable data rights before redistributing data or using this application commercially; replace alpha_trend/data.py with a licensed provider adapter where necessary.

Market data may be delayed, incomplete, revised, rate limited, or unavailable for a ticker. The app caches downloads for one hour. A historical test does not establish that a model will perform in live markets. This software is for research and education, not financial advice.

## Production checklist

- Use a licensed, dependable market-data source for commercial or user-facing production use.
- Lock all transitive Python dependencies and build in a reproducible environment.
- Add authentication, authorization, per-user rate limits, and audit logs before exposing the app to multiple users.
- Persist model/data provenance and evaluation windows; monitor performance and data drift.
- Add richer baselines, transaction costs, corporate-action checks, and strategy-level backtests before considering an execution workflow.
- Do not connect the signals directly to a brokerage or automated trading system without independent risk controls and validation.
