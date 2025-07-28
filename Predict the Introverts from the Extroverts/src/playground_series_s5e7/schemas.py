"""DataFrame models and schemas."""

from dataclasses import asdict, dataclass
from typing import ClassVar

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Series

ALLOWED_ANSWERS = ["Yes", "No"]
ALLOWED_PERSONALITIES = ["Introvert", "Extrovert"]


class FeaturesBase(pa.DataFrameModel):
    """Features available from the base data."""

    id: Series[pa.dtypes.Int32] = pa.Field(
        ge=0, nullable=False, unique=True, description="Unique identifier.", coerce=True
    )
    time_spent_alone: Series[pd.Int16Dtype] = pa.Field(
        ge=0,
        le=11,
        nullable=True,
        description="Hours spent alone daily.",
        coerce=True,
    )
    stage_fear: Series[str] = pa.Field(
        nullable=True,
        isin=ALLOWED_ANSWERS,
        description="Presence of stage fright.",
        coerce=True,
    )
    social_event_attendance: Series[pd.Int16Dtype] = pa.Field(
        ge=0,
        le=10,
        nullable=True,
        description="Frequency of social events.",
        coerce=True,
    )
    going_outside: Series[pd.Int16Dtype] = pa.Field(
        ge=0,
        le=7,
        nullable=True,
        description="Frequency of going outside.",
        coerce=True,
    )
    drained_after_socializing: Series[str] = pa.Field(
        nullable=True,
        isin=ALLOWED_ANSWERS,
        description="Feeling drained after socializing.",
        coerce=True,
    )
    friends_circle_size: Series[pd.Int16Dtype] = pa.Field(
        ge=0,
        le=15,
        nullable=True,
        description="Number of close friends.",
        coerce=True,
    )
    post_frequency: Series[pd.Int16Dtype] = pa.Field(
        ge=0,
        le=10,
        nullable=True,
        description="Social media post frequency.",
        coerce=True,
    )


class RawTarget(FeaturesBase):
    """Raw features with target."""

    personality: Series[str] = pa.Field(
        nullable=False,
        description="Target variable.",
        isin=ALLOWED_PERSONALITIES,
        coerce=True,
    )


class EngineeredFeatures(pa.DataFrameModel):
    """Engineered features."""

    id: Series[pa.dtypes.Int32] = pa.Field(
        ge=0, nullable=False, unique=True, description="Unique identifier."
    )
    time_spent_alone: Series[pa.dtypes.Float32] = pa.Field(
        ge=0,
        le=1,
        nullable=False,
        description="Min-max normalized time spent alone.",
    )
    social_event_attendance: Series[pa.dtypes.Float32] = pa.Field(
        ge=0,
        le=1,
        nullable=False,
        description="Min-max normalized frequency of social events.",
    )
    going_outside: Series[pa.dtypes.Float32] = pa.Field(
        ge=0,
        le=1,
        nullable=False,
        description="Min-max normalized frequency of going outside.",
    )
    friends_circle_size: Series[pa.dtypes.Float32] = pa.Field(
        ge=0,
        le=1,
        nullable=False,
        description="Min-max normalized number of close friends.",
    )
    post_frequency: Series[pa.dtypes.Float32] = pa.Field(
        ge=0,
        le=1,
        nullable=False,
        description="Min-max normalized social media post frequency.",
    )


@dataclass(frozen=True)
class EngineeredStatistics:
    """Statistics about the min-max normalized samples.

    :param count: total number of samples.
    :param count_<feature>: non-null count.
    :param mean_<feature>: non-null sample mean.
    """

    count: int
    count_time_spent_alone: int
    mean_time_spent_alone: float
    count_social_event_attendance: int
    mean_social_event_attendance: float
    count_going_outside: int
    mean_going_outside: float
    count_friends_circle_size: int
    mean_friends_circle_size: float
    count_post_frequency: int
    mean_post_frequency: float

    def asdict(self) -> dict:
        return asdict(self)


class EngineeredTarget(EngineeredFeatures):
    """Engineered features with target."""

    personality: Series[pa.dtypes.Int8] = pa.Field(
        nullable=False,
        description="Encoded target variable (0=Introvert, 1=Extrovert).",
        isin=[0, 1],
        coerce=True,
    )


class Meta:
    """Meta information about schemas.

    Allows the user to read group of related variables
    with a single name.
    """

    class FeaturesBase:
        FEATURES_COLUMNS: ClassVar[list[str]] = [
            FeaturesBase.time_spent_alone,
            FeaturesBase.stage_fear,
            FeaturesBase.social_event_attendance,
            FeaturesBase.going_outside,
            FeaturesBase.drained_after_socializing,
            FeaturesBase.friends_circle_size,
            FeaturesBase.post_frequency,
        ]

        CATEGORICAL_FEATURES_COLUMNS: ClassVar[list[str]] = [
            FeaturesBase.stage_fear,
            FeaturesBase.drained_after_socializing,
        ]

    class RawTarget(FeaturesBase):
        pass

    class EngineeredFeatures:
        FEATURES_COLUMNS: ClassVar[list[str]] = [
            EngineeredFeatures.time_spent_alone,
            EngineeredFeatures.social_event_attendance,
            EngineeredFeatures.going_outside,
            EngineeredFeatures.friends_circle_size,
            EngineeredFeatures.post_frequency,
        ]

        FEATURES_MIN_MAX: ClassVar[list[str, tuple[int, int]]] = list(
            zip(
                FEATURES_COLUMNS,
                [(0, 11), (0, 11), (0, 7), (0, 15), (0, 10)],
            )
        )

    class EngineeredTarget(EngineeredFeatures):
        pass
