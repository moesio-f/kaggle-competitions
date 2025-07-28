"""CLI for feature engineering, training
and inference.
"""

import json
import logging
from pathlib import Path

import click
import joblib
import pandas as pd

from . import feature_engineering, inference, schemas, training

LOGGER = logging.getLogger(__name__)

_DEFAULT_KWARGS = {"context_settings": {"show_default": True}}


@click.group()
def entrypoint():
    pass


@click.command(
    "prepare_train_data", **_DEFAULT_KWARGS, help="Prepare train/test datasets."
)
@click.option(
    "--source", type=Path, required=True, help="Path to source (raw) data for training."
)
@click.option(
    "--output",
    type=Path,
    required=True,
    help="Output directory for processed training data.",
)
@click.option(
    "--frac", type=float, default=0.8, help="Training fraction to split samples."
)
@click.option("--seed", type=int, default=42, help="Random seed.")
def prepare_train_data(source: Path, output: Path, frac: float, seed: int):
    assert source.is_file(), "Source must exist and be a CSV."
    output.mkdir(exist_ok=True, parents=True)

    LOGGER.info("Loading raw data...")
    raw = schemas.RawTarget.validate(pd.read_csv(source).rename(columns=str.lower))
    LOGGER.info("Loaded raw data: %d samples, %d columns.", len(raw), len(raw.columns))

    LOGGER.info("Preparing train/test data...")
    train, test, statistics = feature_engineering.prepare_train_data(raw, frac, seed)
    LOGGER.info(
        "Data prepared. Train has %d samples, test has %d samples.",
        len(train),
        len(test),
    )

    LOGGER.info("Saving datasets and statistics...")
    train.to_parquet(output.joinpath("train.parquet"))
    test.to_parquet(output.joinpath("test.parquet"))
    with output.joinpath("train_statistics.json").open("w+") as f:
        json.dump(statistics.asdict(), f, indent=2, ensure_ascii=False)
    LOGGER.info("Done!")


@click.command(
    "process_batch", **_DEFAULT_KWARGS, help="Process raw features for inference."
)
@click.option(
    "--source",
    type=Path,
    required=True,
    help="Path to source (raw) data for inference.",
)
@click.option(
    "--statistics",
    type=Path,
    required=True,
    help="Path to train statistics for data imputation.",
)
@click.option(
    "--output-file",
    type=Path,
    required=True,
    help="Output file for inference data.",
)
def process_batch(source: Path, statistics: Path, output_file: Path):
    assert source.exists() and source.is_file(), "Source must exist and be a CSV."
    assert statistics.is_file(), "Statistics must exist and be a JSON."
    assert output_file.name.endswith(".parquet"), "Output file should be parquet."
    output_file.parent.mkdir(exist_ok=True, parents=True)

    LOGGER.info("Loading sources...")
    with statistics.open("r") as f:
        statistics = schemas.EngineeredStatistics(**json.load(f))
    df = schemas.FeaturesBase.validate(pd.read_csv(source).rename(columns=str.lower))
    LOGGER.info("Sources loaded. DataFrame has %d samples.", len(df))

    LOGGER.info("Processing batch...")
    df = feature_engineering.process_batch(df, statistics)
    df.to_parquet(output_file)
    LOGGER.info("Batch processed and saved to '%s'.", output_file)


@click.command(
    "train_finetuned_model", **_DEFAULT_KWARGS, help="Train and finetune a SVM."
)
@click.option(
    "--train", type=Path, required=True, help="Path to processed training data."
)
@click.option("--test", type=Path, required=True, help="Path to processed test data.")
@click.option("--output", type=Path, required=True, help="Output directory for model.")
@click.option("--seed", type=int, default=42, help="Random seed for fine-tuning.")
@click.option(
    "--trials",
    type=int,
    default=100,
    help="Fine-tuning trials (i.e., optimization steps).",
)
@click.option(
    "--disable-best-model-train-on-full",
    is_flag=True,
    help="Disable training best model on full labeled dataset (train+test)."
    "Best model configuration is always chosen by training on `train` and "
    "evaluating on `test`.",
)
def train_finetuned_model(
    train: Path,
    test: Path,
    output: Path,
    seed: int,
    trials: int,
    disable_best_model_train_on_full: bool,
):
    assert trials > 0, "Trials must be positive."
    output.mkdir(exist_ok=True, parents=True)
    train = pd.read_parquet(train)
    test = pd.read_parquet(test)

    LOGGER.info("Starting model fine-tuning...")
    train_best_model_on_full = not disable_best_model_train_on_full
    model = training.train_holdout_finetuned(
        train,
        test,
        seed=seed,
        n_trials=trials,
        train_best_model_on_full=train_best_model_on_full,
    )

    LOGGER.info("Fine-tune complete. Best model ACC: %.5f.", model.accuracy)
    data = model.asdict()
    data["seed"] = seed
    data["n_trials"] = trials
    data["trained_on_full"] = train_best_model_on_full
    model = data.pop("model")
    joblib.dump(model, output.joinpath("svm.joblib"))
    with output.joinpath("metadata.json").open("w+") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    LOGGER.info("Model and metadata available at '%s'.", output)


@click.command(
    "run_inference", **_DEFAULT_KWARGS, help="Run inference with a trained SVM."
)
@click.option("--df", type=Path, required=True, help="Path to raw features.")
@click.option(
    "--statistics",
    type=Path,
    required=True,
    help="Path to train statistics for data imputation.",
)
@click.option(
    "--model",
    type=Path,
    required=True,
    help="Path to model directory (pickle+metadata).",
)
@click.option(
    "--output-file", type=Path, required=True, help="Output file to save predictions."
)
def run_inference(df: Path, statistics: Path, model: Path, output_file: Path):
    assert df.is_file(), "Features must exist and be a CSV."
    assert statistics.is_file(), "Statistics must exist and be a JSON."
    assert model.is_dir(), "Model directory must exist."
    output_file.parent.mkdir(exist_ok=True, parents=True)

    # Loading
    LOGGER.info("Loading sources...")
    with statistics.open("r") as f:
        statistics = schemas.EngineeredStatistics(**json.load(f))
    df = schemas.FeaturesBase.validate(pd.read_csv(df).rename(columns=str.lower))
    model = inference.Model(model)
    LOGGER.info(
        "Sources loaded. DataFrame has %d samples. Model had %.3f ACC in training.",
        len(df),
        model.metadata["accuracy"],
    )

    # Online feature engineering
    LOGGER.info("Running online feature engineering...")
    df = feature_engineering.process_batch(df, statistics)
    LOGGER.info("Data is ready for inference.")

    # Model prediction
    LOGGER.info("Running predictions...")
    preds = model.predict(df)
    preds = preds[
        [schemas.EngineeredTarget.id, schemas.EngineeredTarget.personality]
    ].rename(columns={"personality": "target"})

    # Inverse map personality
    preds[schemas.EngineeredTarget.personality] = preds["target"].map(
        {1: "Extrovert", 0: "Introvert"}
    )
    preds = preds.drop(columns="target")

    # Save predictions
    if output_file.name.endswith(".parquet"):
        preds.to_parquet(output_file)
    else:
        preds.to_csv(output_file, index=False)
    LOGGER.info("Predictions available at '%s'.", output_file)


# Add commands to entrypoint
entrypoint.add_command(prepare_train_data)
entrypoint.add_command(process_batch)
entrypoint.add_command(train_finetuned_model)
entrypoint.add_command(run_inference)

if __name__ == "__main__":
    entrypoint()
