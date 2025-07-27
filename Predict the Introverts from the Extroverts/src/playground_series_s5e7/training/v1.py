"""v1 SVM for classification."""

import logging
from dataclasses import asdict, dataclass

import numpy as np
import optuna
import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame
from sklearn.metrics import accuracy_score as acc_score
from sklearn.svm import SVC

from playground_series_s5e7.schemas import V1Target

LOGGER = logging.getLogger(__name__)
_FEATURES_COLUMNS: list[str] = [
    "time_spent_alone",
    "social_event_attendance",
    "going_outside",
    "friends_circle_size",
    "post_frequency",
]


@dataclass(frozen=True)
class ScoredModel:
    model: SVC
    accuracy: float
    C: float
    kernel: str
    shrinking: bool
    class_weight: str | dict | None
    random_state: int
    degree: int = 3
    gamma: str = "scale"
    coef0: float = 0.0

    def asdict(self) -> dict:
        return asdict(self)


@pa.check_input(V1Target)
def train_model(
    train: DataFrame[V1Target],
    C: float = 1.0,
    kernel: str = "rbf",
    degree: int = 3,
    gamma: str = "scale",
    coef0: float = 0.0,
    shrinking: bool = True,
    class_weight: str | dict | None = None,
    random_state: int | None = None,
) -> SVC:
    X = train[_FEATURES_COLUMNS].values
    y = train.personality.values
    model = SVC(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        shrinking=shrinking,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


@pa.check_input(V1Target, "test")
def accuracy_score(model: SVC, test: DataFrame[V1Target]) -> float:
    """Compute the accuracy score. The model is assumed
    to come from `train_model` or `train_holdout_finetuned`.

    :param model: SVC.
    :param test: test DataFrame.
    :return: accuracy.
    """
    return acc_score(test.personality, model.predict(test[_FEATURES_COLUMNS].values))


@pa.check_input(V1Target, "train")
@pa.check_input(V1Target, "test")
def train_holdout_finetuned(
    train: DataFrame[V1Target],
    test: DataFrame[V1Target],
    seed: int,
    n_trials: int = 100,
    train_best_model_on_full: bool = True,
    C_space: tuple[float, float] = (1e-3, 1e1),
    kernel_choices: tuple[str] = ("linear", "poly", "rbf", "sigmoid"),
    degree_space: tuple[int, int] = (1, 3),
    gamma_choices: tuple[str | float] = ("scale", "auto"),
    coef0_space: tuple[float, float] = (0.0, 2e1),
    shrinking_choices: tuple[bool] = (True, False),
    class_weight_choices: tuple[dict | str | None] = ("balanced", None),
) -> ScoredModel:
    """Return a fine-tuned model using a hold-out evaluation
    to infer performance.

    :param train: train set.
    :param test: test set.
    :param seed: random seed for experiments.
    :param n_trials: number of optimization steps.
    :param train_best_model_on_full: if the final model should
        be trained on train+test (true) or only train (false).
    :param C_space: space for C parameter.
    :param kernel_choices: possible kernels to use.
    :param degree_space: space for polynomial degree (only
        used for "poly" kernel).
    :param gamma_choices: possible values for gamma (only
        used for non-linear kernels).
    :param coef0_space: coef0 space (only used for "poly" or
        "sigmoid" kernels).
    :param shrinking_choices: possibles values for shrinking.
    :param class_weight_choices: possible values for class_weight.
    :return: model with highest accuracy.
    """
    # Initialize RNG for fine-tuning
    rng = np.random.default_rng(seed)

    def gen_seed() -> int:
        return rng.integers(low=0, high=99999).item()

    def objective(trial: optuna.Trial) -> float:
        """Score for a SVM model.

        :param trial: trial objective.
        :return: accuracy score (higher=better).
        """
        C = trial.suggest_float("C", *C_space)
        shrinking = trial.suggest_categorical("shrinking", shrinking_choices)
        kernel = trial.suggest_categorical("kernel", kernel_choices)
        class_weight = trial.suggest_categorical("class_weight", class_weight_choices)

        # Conditions for other values
        is_poly = kernel == "poly"
        is_sigmoid = kernel == "sigmoid"
        is_linear = kernel == "linear"

        # Conditional parameters
        degree = trial.suggest_int("degree", *degree_space) if is_poly else 3
        gamma = (
            trial.suggest_categorical("gamma", gamma_choices)
            if not is_linear
            else "scale"
        )
        coef0 = (
            trial.suggest_float("coef0", *coef0_space) if is_poly or is_sigmoid else 0.0
        )

        # Train and score model
        random_state = gen_seed()
        params = dict(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            class_weight=class_weight,
        )
        LOGGER.info("Training model with parameters: %s", params)
        model = train_model(
            train,
            **params,
            random_state=random_state,
        )
        score = accuracy_score(model, test)
        trial.set_user_attr("random_state", random_state)
        return score

    # Setup optuna
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=gen_seed()),
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
    )

    # Find best trial
    best_trial = study.best_trial

    # Get best model and re-train
    df = train
    if train_best_model_on_full:
        df = pd.concat([train, test], ignore_index=True)

    params = best_trial.params
    model = train_model(df, **params)

    # Return scored model
    return ScoredModel(
        model=model, accuracy=best_trial.value, **params, **best_trial.user_attrs
    )
