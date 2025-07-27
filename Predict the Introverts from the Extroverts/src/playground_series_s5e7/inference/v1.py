"""v1 SVM for classification."""

from pathlib import Path
import json
import joblib
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame

from playground_series_s5e7.schemas import V1Features, V1Target


class ModelV1:
    _FEATURES_COLUMNS: list[str] = [
        "time_spent_alone",
        "social_event_attendance",
        "going_outside",
        "friends_circle_size",
        "post_frequency",
    ]

    def __init__(self, model_directory: Path):
        assert model_directory.exists(), "Model directory doesn't exist."
        models = list(model_directory.glob("*.joblib"))
        metadata = list(model_directory.glob("metadata.json"))
        if len(models) != 1 or len(metadata) != 1:
            raise ValueError("Couldn't find model in target directory. ")
        self.model = joblib.load(models[0])
        self.metadata = json.loads(metadata[0].read_text())

    @pa.check_input(V1Features)
    def predict(self, df: DataFrame[V1Features]) -> DataFrame[V1Target]:
        """Run predictions.

        :param df: inference DataFrame.
        :return: predictions.
        """
        df = df.copy()
        df["personality"] = self.model.predict(df[self._FEATURES_COLUMNS].values)
        return V1Target.validate(df)
