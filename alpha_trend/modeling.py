"""Leakage-aware walk-forward evaluation and next-session forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor

from alpha_trend.config import MODEL_CONFIG, ModelConfig
from alpha_trend.features import FEATURE_COLUMNS, build_features


class ModelingError(RuntimeError):
    """Raised when there is not enough valid history to train and evaluate."""


@dataclass(frozen=True)
class Metrics:
    rmse: float
    mae: float
    r2: float
    directional_accuracy: float


@dataclass(frozen=True)
class Forecast:
    last_close: float
    predicted_close: float
    expected_return: float
    direction: str
    probability_up: float
    confidence: float
    signal: str
    latest_data_date: pd.Timestamp


@dataclass(frozen=True)
class ModelEvaluation:
    walk_forward: Metrics
    holdout: Metrics
    test_predictions: pd.DataFrame
    forecast: Forecast
    cv_folds: int
    training_rows: int
    test_rows: int


def _make_regressor(config: ModelConfig) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=1.5,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbosity=0,
    )


def _make_classifier(config: ModelConfig) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=1.5,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbosity=0,
    )


def _score(
    actual_price: np.ndarray,
    predicted_price: np.ndarray,
    actual_direction: np.ndarray,
    predicted_direction: np.ndarray,
) -> Metrics:
    return Metrics(
        rmse=sqrt(mean_squared_error(actual_price, predicted_price)),
        mae=mean_absolute_error(actual_price, predicted_price),
        r2=r2_score(actual_price, predicted_price) if len(actual_price) > 1 else float("nan"),
        directional_accuracy=accuracy_score(actual_direction, predicted_direction),
    )


def _supervised_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    frame["target_next_close"] = frame["Close"].shift(-1)
    next_close = frame["target_next_close"]
    frame["target_direction"] = np.where(
        next_close.notna(), (next_close > frame["Close"]).astype("int8"), np.nan
    )
    frame["target_date"] = pd.Series(frame.index, index=frame.index).shift(-1)
    required = [*FEATURE_COLUMNS, "target_next_close", "target_direction", "target_date"]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    frame["target_direction"] = frame["target_direction"].astype("int8")
    return frame


def _signal(
    probability_up: float,
    expected_return: float,
    probability_threshold: float,
    minimum_move: float,
) -> str:
    if probability_up >= probability_threshold and expected_return >= minimum_move:
        return "BUY"
    if (
        probability_up <= 1 - probability_threshold
        and expected_return <= -minimum_move
    ):
        return "SELL"
    return "HOLD"


def evaluate_and_forecast(
    ohlcv: pd.DataFrame,
    *,
    config: ModelConfig = MODEL_CONFIG,
    probability_threshold: float | None = None,
    minimum_move: float | None = None,
) -> ModelEvaluation:
    """Evaluate on chronological folds and a separated holdout, then fit a live forecast.

    The one-row gap prevents the final training label (next session's close) from
    using a price belonging to the first validation or holdout feature date.
    Every scaler is fitted only on its corresponding training observations.
    """
    features = build_features(ohlcv)
    supervised = _supervised_frame(features)

    if len(supervised) < config.minimum_samples:
        raise ModelingError(
            f"Only {len(supervised)} usable rows remain after indicator warm-up. "
            f"Choose a longer date range; at least {config.minimum_samples} are required."
        )

    split_index = int(len(supervised) * (1 - config.test_fraction))
    train_stop = split_index - config.gap
    train_frame = supervised.iloc[:train_stop].copy()
    test_frame = supervised.iloc[split_index:].copy()

    if len(train_frame) < config.cv_splits + 2 or test_frame.empty:
        raise ModelingError("The selected date range is too short for a holdout evaluation.")
    if train_frame["target_direction"].nunique() < 2:
        raise ModelingError("Training history contains only one direction class; expand the date range.")

    x_train_all = train_frame[FEATURE_COLUMNS]
    y_train_price = train_frame["target_next_close"].to_numpy(dtype="float64")
    y_train_direction = train_frame["target_direction"].to_numpy(dtype="int8")

    cv_actual_price: list[np.ndarray] = []
    cv_predicted_price: list[np.ndarray] = []
    cv_actual_direction: list[np.ndarray] = []
    cv_predicted_direction: list[np.ndarray] = []
    completed_folds = 0

    splitter = TimeSeriesSplit(n_splits=config.cv_splits, gap=config.gap)
    for fold_train_idx, fold_valid_idx in splitter.split(x_train_all):
        fold_x_train = x_train_all.iloc[fold_train_idx]
        fold_y_price = y_train_price[fold_train_idx]
        fold_y_direction = y_train_direction[fold_train_idx]
        if np.unique(fold_y_direction).size < 2:
            continue

        fold_x_valid = x_train_all.iloc[fold_valid_idx]
        scaler = MinMaxScaler()
        scaled_fold_train = scaler.fit_transform(fold_x_train)
        scaled_fold_valid = scaler.transform(fold_x_valid)

        regressor = _make_regressor(config)
        classifier = _make_classifier(config)
        regressor.fit(scaled_fold_train, fold_y_price)
        classifier.fit(scaled_fold_train, fold_y_direction)

        cv_actual_price.append(
            train_frame["target_next_close"].iloc[fold_valid_idx].to_numpy(dtype="float64")
        )
        cv_predicted_price.append(regressor.predict(scaled_fold_valid))
        cv_actual_direction.append(
            train_frame["target_direction"].iloc[fold_valid_idx].to_numpy(dtype="int8")
        )
        cv_predicted_direction.append(classifier.predict(scaled_fold_valid).astype("int8"))
        completed_folds += 1

    if completed_folds == 0:
        raise ModelingError(
            "Walk-forward folds could not be trained. Select a longer date range with both up and down days."
        )

    walk_forward_metrics = _score(
        np.concatenate(cv_actual_price),
        np.concatenate(cv_predicted_price),
        np.concatenate(cv_actual_direction),
        np.concatenate(cv_predicted_direction),
    )

    # Final holdout scaler and models see training rows only; the gap row is excluded.
    test_scaler = MinMaxScaler()
    scaled_train = test_scaler.fit_transform(x_train_all)
    scaled_test = test_scaler.transform(test_frame[FEATURE_COLUMNS])

    test_regressor = _make_regressor(config)
    test_classifier = _make_classifier(config)
    test_regressor.fit(scaled_train, y_train_price)
    test_classifier.fit(scaled_train, y_train_direction)
    predicted_test_price = test_regressor.predict(scaled_test)
    predicted_test_direction = test_classifier.predict(scaled_test).astype("int8")

    actual_test_price = test_frame["target_next_close"].to_numpy(dtype="float64")
    actual_test_direction = test_frame["target_direction"].to_numpy(dtype="int8")
    holdout_metrics = _score(
        actual_test_price,
        predicted_test_price,
        actual_test_direction,
        predicted_test_direction,
    )

    test_predictions = pd.DataFrame(
        {
            "actual_close": actual_test_price,
            "predicted_close": predicted_test_price,
            "actual_direction": actual_test_direction,
            "predicted_direction": predicted_test_direction,
        },
        index=pd.DatetimeIndex(test_frame["target_date"], name="Date"),
    )
    test_predictions = test_predictions.sort_index()

    # Refit on all labeled observations. The final market row is still unlabeled
    # and is used only as the live next-session feature vector.
    final_scaler = MinMaxScaler()
    scaled_history = final_scaler.fit_transform(supervised[FEATURE_COLUMNS])
    final_regressor = _make_regressor(config)
    final_classifier = _make_classifier(config)
    final_regressor.fit(scaled_history, supervised["target_next_close"].to_numpy(dtype="float64"))
    final_classifier.fit(
        scaled_history, supervised["target_direction"].to_numpy(dtype="int8")
    )

    latest_features = features.dropna(subset=FEATURE_COLUMNS).iloc[[-1]]
    if latest_features.empty:
        raise ModelingError("The latest price row does not have complete model features.")

    scaled_latest = final_scaler.transform(latest_features[FEATURE_COLUMNS])
    predicted_close = float(final_regressor.predict(scaled_latest)[0])
    latest_close = float(latest_features["Close"].iloc[0])
    predicted_direction = int(final_classifier.predict(scaled_latest)[0])
    probability_columns = list(final_classifier.classes_)
    probability_up = float(
        final_classifier.predict_proba(scaled_latest)[0][probability_columns.index(1)]
    )
    confidence = max(probability_up, 1 - probability_up)
    expected_return = predicted_close / latest_close - 1

    forecast = Forecast(
        last_close=latest_close,
        predicted_close=predicted_close,
        expected_return=expected_return,
        direction="Up" if predicted_direction == 1 else "Down",
        probability_up=probability_up,
        confidence=confidence,
        signal=_signal(
            probability_up,
            expected_return,
            probability_threshold
            if probability_threshold is not None
            else config.default_signal_probability,
            minimum_move if minimum_move is not None else config.default_minimum_move,
        ),
        latest_data_date=pd.Timestamp(latest_features.index[-1]),
    )

    return ModelEvaluation(
        walk_forward=walk_forward_metrics,
        holdout=holdout_metrics,
        test_predictions=test_predictions,
        forecast=forecast,
        cv_folds=completed_folds,
        training_rows=len(train_frame),
        test_rows=len(test_frame),
    )
