"""Application and model defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Reproducible model and evaluation settings."""

    test_fraction: float = 0.20
    cv_splits: int = 5
    minimum_samples: int = 250
    gap: int = 1
    random_state: int = 42
    n_estimators: int = 250
    max_depth: int = 3
    learning_rate: float = 0.035
    n_jobs: int = 2
    default_signal_probability: float = 0.60
    default_minimum_move: float = 0.005


MODEL_CONFIG = ModelConfig()
DATA_CACHE_TTL_SECONDS = 60 * 60
MAX_TICKER_LENGTH = 32
