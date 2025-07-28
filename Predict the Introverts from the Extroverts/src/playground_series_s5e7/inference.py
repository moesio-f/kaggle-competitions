"""Model inference."""

import json
from pathlib import Path

import joblib
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame

from playground_series_s5e7.schemas import EngineeredFeatures, EngineeredTarget, Meta


class Model:
    """Trained model. Allows to run
    inferences on the engineered features.

    :param model: underlying scikit model.
    :param metadata: model metadata.
    """

    def __init__(self, model_directory: Path):
        """Constructor.

        :param model_directory: directory where model is stored.
        """
        assert model_directory.exists(), "Model directory doesn't exist."
        models = list(model_directory.glob("*.joblib"))
        metadata = list(model_directory.glob("metadata.json"))
        if len(models) != 1 or len(metadata) != 1:
            raise ValueError("Couldn't find model in target directory. ")
        self.model = joblib.load(models[0])
        self.metadata = json.loads(metadata[0].read_text())

    @pa.check_input(EngineeredFeatures)
    def predict(self, df: DataFrame[EngineeredFeatures]) -> DataFrame[EngineeredTarget]:
        """Run predictions.

        :param df: inference DataFrame.
        :return: predictions.
        """
        df = df.copy()
        df["personality"] = self.model.predict(
            df[Meta.EngineeredFeatures.FEATURES_COLUMNS].values
        )
        return EngineeredTarget.validate(df)
