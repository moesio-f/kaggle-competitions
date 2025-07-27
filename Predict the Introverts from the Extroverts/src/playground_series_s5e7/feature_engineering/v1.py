"""v1 feature engineering pipelines."""

import pandera.pandas as pa
from pandera.typing.pandas import DataFrame

from playground_series_s5e7.schemas import (
    FeaturesBase,
    RawTarget,
    V1Features,
    V1FeaturesStatistics,
    V1Target,
)

_FEATURES_COLUMNS: list[str] = [
    "time_spent_alone",
    "social_event_attendance",
    "going_outside",
    "friends_circle_size",
    "post_frequency",
]
_FEATURES_MIN_MAX: list[tuple[str, tuple[int, int]]] = list(
    zip(
        _FEATURES_COLUMNS,
        [(0, 11), (0, 11), (0, 7), (0, 15), (0, 10)],
    )
)


@pa.check_input(RawTarget)
def prepare_train_data(
    df: DataFrame[RawTarget],
    frac_train: float = 0.8,
    seed: int = 42,
) -> tuple[DataFrame[V1Target], DataFrame[V1Target], V1FeaturesStatistics]:
    """Prepare train, test and train statistics from a
    labeled database.

    :param df: labeled feature DataFrame.
    :param frac_train: fraction of samples to use for training.
    :param seed: random seed.
    :return: (train, test, statistics).
    """
    assert frac_train > 0.0 and frac_train <= 1.0, "Train fraction should be in [0, 1]."

    # Remove redundant features
    df = df.drop(columns=["drained_after_socializing", "stage_fear"])

    df["personality"] = df["personality"].map({"Extrovert": 1, "Introvert": 0})

    # Min-max normalization
    for col, (min, max) in _FEATURES_MIN_MAX:
        s = (df[col] - min) / (max - min)
        df[col] = s.astype("float32")

    # Train and test sets
    train = df.sample(frac=frac_train, random_state=seed)
    test = df[~df.id.isin(train.id)]

    # Gather summary statistics
    stats = dict(count=len(train))
    for col in _FEATURES_COLUMNS:
        s = train[col]
        stats[f"count_{col}"] = stats["count"] - s.isna().sum().item()
        stats[f"mean_{col}"] = s.dropna().mean().item()
    stats = V1FeaturesStatistics(**stats)

    # Data imputation
    for col in _FEATURES_COLUMNS:
        fill_value = getattr(stats, f"mean_{col}")
        train.loc[:, col] = train[col].fillna(value=fill_value)
        test.loc[:, col] = test[col].fillna(value=fill_value)

    return V1Target.validate(train), V1Target.validate(test), stats


@pa.check_input(FeaturesBase)
def process_batch(
    df: DataFrame[FeaturesBase],
    statistics: V1FeaturesStatistics,
) -> DataFrame[V1Features]:
    """Apply feature engineering to the base features.

    :param df: base features.
    :param statistics: training statistics of engineered
        features. Required for data imputation.
    :return: feature engineered features.
    """
    # Redundant features (high correlation between themselves and others)
    df = df.drop(columns=["drained_after_socializing", "stage_fear"])

    # Min-max normalization
    for col, (min, max) in _FEATURES_MIN_MAX:
        df[col] = df[col].astype("float32")
        s = (df[col] - min) / (max - min)
        df.loc[:, col] = s

    # Data imputation
    for col in _FEATURES_COLUMNS:
        fill_value = getattr(statistics, f"mean_{col}")
        df.loc[:, col] = df[col].fillna(value=fill_value)

    return V1Features.validate(df)
