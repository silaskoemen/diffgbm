"""
Contains different score models to be used to approximate the score of a given SDE.
"""

import abc
import warnings
from typing import Any
from typing import cast

import lightgbm as lgb
import numpy as np
from jaxtyping import Float
from jaxtyping import Int
from sklearn.model_selection import train_test_split

from treeffuser.sde import DiffusionSDE

###################################################
# Score parameterizations
###################################################


class ScoreParameterization(abc.ABC):
    """
    Defines the regression target used to train a score model and how model predictions
    are converted back into a score during reverse-time sampling.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def make_target(
        self,
        y0: Float[np.ndarray, "batch y_dim"],
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        z: Float[np.ndarray, "batch y_dim"],
        mean: Float[np.ndarray, "batch y_dim"],
        std: Float[np.ndarray, "batch y_dim"],
        t: Float[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        pass

    @abc.abstractmethod
    def reconstruct_score(
        self,
        prediction: Float[np.ndarray, "batch"],
        perturbed_y: Float[np.ndarray, "batch"],
        std: Float[np.ndarray, "batch"],
        t: Float[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch"]:
        pass


class NoiseParameterization(ScoreParameterization):
    """
    Current Treeffuser behavior: train LightGBM to predict the added negative noise and
    reconstruct the score by dividing the prediction by the perturbation standard deviation.
    """

    @property
    def name(self) -> str:
        return "noise"

    def make_target(
        self,
        y0: Float[np.ndarray, "batch y_dim"],
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        z: Float[np.ndarray, "batch y_dim"],
        mean: Float[np.ndarray, "batch y_dim"],
        std: Float[np.ndarray, "batch y_dim"],
        t: Float[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        return -1.0 * z

    def reconstruct_score(
        self,
        prediction: Float[np.ndarray, "batch"],
        perturbed_y: Float[np.ndarray, "batch"],
        std: Float[np.ndarray, "batch"],
        t: Float[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch"]:
        return prediction / std


def get_score_parameterization(
    parameterization: str | ScoreParameterization,
) -> ScoreParameterization:
    if isinstance(parameterization, ScoreParameterization):
        return parameterization
    if parameterization == "noise":
        return NoiseParameterization()
    raise ValueError(f"Unknown score parameterization: {parameterization}")


###################################################
# Noise feature builders
###################################################


class NoiseFeatureBuilder(abc.ABC):
    """
    Builds the feature matrix passed to the underlying regressor from the perturbed
    sample, side information, and time. Centralizing this lets training and inference
    share one definition so the two paths cannot drift apart.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def make_features(
        self,
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Float[np.ndarray, "batch 1"],
        sde: DiffusionSDE,
    ) -> Float[np.ndarray, "batch feat_dim"]:
        pass


class RawTimeFeatureBuilder(NoiseFeatureBuilder):
    """
    Current Treeffuser feature layout: [perturbed_y, X, t].
    """

    @property
    def name(self) -> str:
        return "raw_time"

    def make_features(
        self,
        perturbed_y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Float[np.ndarray, "batch 1"],
        sde: DiffusionSDE,
    ) -> Float[np.ndarray, "batch feat_dim"]:
        return np.concatenate([perturbed_y, X, t], axis=1)


def get_noise_feature_builder(
    feature_builder: str | NoiseFeatureBuilder,
) -> NoiseFeatureBuilder:
    if isinstance(feature_builder, NoiseFeatureBuilder):
        return feature_builder
    if feature_builder == "raw_time":
        return RawTimeFeatureBuilder()
    raise ValueError(f"Unknown noise feature builder: {feature_builder}")


###################################################
# Helper functions
###################################################


def _fit_one_lgbm_model(
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    X_val: Float[np.ndarray, "batch x_dim"] | None,
    y_val: Float[np.ndarray, "batch y_dim"] | None,
    seed: int | None,
    verbose: int,
    cat_idx: list[int] | None = None,
    n_jobs: int | None = -1,
    early_stopping_rounds: int | None = None,
    **lgbm_args,
) -> lgb.LGBMRegressor:
    """
    Simple wrapper for fitting a lightgbm model. See
    the lightgbm score function documentation for more details.
    """
    callbacks = None
    if early_stopping_rounds is not None:
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=verbose > 0)]

    model = lgb.LGBMRegressor(
        random_state=seed,
        verbose=verbose,
        n_jobs=n_jobs,
        linear_tree=False,
        **lgbm_args,
    )
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]
    else:
        eval_set = None
    categorical_feature: list[int] | str = "auto" if cat_idx is None else cat_idx
    model.fit(
        X=X,
        y=y,
        eval_set=cast(Any, eval_set),
        callbacks=cast(Any, callbacks),
        categorical_feature=categorical_feature,
    )
    return model


def _make_training_data(
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    sde: DiffusionSDE,
    n_repeats: int | None,
    eval_percent: float | None,
    cat_idx: list[int] | None = None,
    seed: int | None = None,
    score_parameterization: ScoreParameterization | None = None,
    noise_feature_builder: NoiseFeatureBuilder | None = None,
):
    """
    Creates the training data for the score model. This functions assumes that
    1.  Score is parametrized as score(y, x, t) = GBT(y, x, t) / std(t)
    2.  The loss that we want to use is
        || std(t) * score(y_perturbed, x, t) - (mean(y, t) - y_perturbed)/std(t) ||^2
        Which corresponds to the standard denoising objective with weights std(t)**2
        This ends up meaning that we optimize
        || GBT(y_perturbed, x, t) - (-z)||^2
        where z is the noise added to y_perturbed.

    Returns:
    - predictors_train: X_train=[y_perturbed_train, x_train, t_train] for lgbm
    - predictors_val: X_val=[y_perturbed_val, x_val, t_val] for lgbm
    - predicted_train: y_train=[-z_train] for lgbm
    - predicted_val: y_val=[-z_val] for lgbm
    """
    if score_parameterization is None:
        score_parameterization = NoiseParameterization()
    if noise_feature_builder is None:
        noise_feature_builder = RawTimeFeatureBuilder()
    EPS = 1e-5  # smallest step we can sample from
    T = sde.T
    rng = np.random.default_rng(seed)

    X_train, X_test, y_train, y_test = X, None, y, None
    predictors_train, predictors_val = None, None
    predicted_train, predicted_val = None, None

    if eval_percent is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=eval_percent, random_state=seed)

    # TRAINING DATA
    n_reps = n_repeats if n_repeats is not None else 1
    X_train = np.tile(X_train, (n_reps, 1))
    y_train = np.tile(y_train, (n_reps, 1))
    t_train = rng.uniform(0, 1, size=(y_train.shape[0], 1)) * (T - EPS) + EPS
    z_train = rng.normal(size=y_train.shape)

    train_mean, train_std = sde.get_mean_std_pt_given_y0(y_train, t_train)
    perturbed_y_train = train_mean + train_std * z_train
    predictors_train = noise_feature_builder.make_features(
        perturbed_y=perturbed_y_train,
        X=X_train,
        t=t_train,
        sde=sde,
    )
    predicted_train = score_parameterization.make_target(
        y0=y_train,
        perturbed_y=perturbed_y_train,
        z=z_train,
        mean=train_mean,
        std=train_std,
        t=t_train,
    )

    # VALIDATION DATA
    if eval_percent is not None:
        assert y_test is not None
        assert X_test is not None
        t_val = rng.uniform(0, 1, size=(y_test.shape[0], 1)) * (T - EPS) + EPS
        z_val = rng.normal(size=(y_test.shape[0], y_test.shape[1]))

        val_mean, val_std = sde.get_mean_std_pt_given_y0(y_test, t_val)
        perturbed_y_val = val_mean + val_std * z_val
        predictors_val = noise_feature_builder.make_features(
            perturbed_y=perturbed_y_val,
            X=X_test,
            t=t_val,
            sde=sde,
        )
        predicted_val = score_parameterization.make_target(
            y0=y_test,
            perturbed_y=perturbed_y_val,
            z=z_val,
            mean=val_mean,
            std=val_std,
            t=t_val,
        )

    cat_idx = [c + y_train.shape[1] for c in cat_idx] if cat_idx is not None else None

    return predictors_train, predictors_val, predicted_train, predicted_val, cat_idx


###################################################
# Score models
###################################################


class ScoreModel(abc.ABC):
    @abc.abstractmethod
    def score(
        self,
        y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Int[np.ndarray, "batch 1"],
    ):

        pass

    @abc.abstractmethod
    def fit(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        sde: DiffusionSDE,
        cat_idx: list[int] | None = None,
    ):
        pass


class LightGBMScoreModel(ScoreModel):
    """
    A score model that uses a LightGBM model (trees) to approximate the score of a given SDE.

    Parameters
    ----------
    n_repeats : int
        How many times to repeat the training dataset when fitting the score. That is, how many
        noisy versions of a point to generate for training.
    eval_percent : float
        Percentage of the training data to use for validation for optional early stopping. It is
        ignored if `early_stopping_rounds` is not set in the `lgbm_args`.
    n_jobs : int
        LightGBM: Number of parallel threads. If set to -1, the number is set to the number of available cores.
    seed : int
        Random seed for generating the training data and fitting the model.
    verbose : int
        Verbosity of the score model.
    **lgbm_args
        Additional arguments to pass to the LightGBM model. See the LightGBM documentation for more
        information. E.g. `early_stopping_rounds`, `n_estimators`, `learning_rate`, `max_depth`,

    Attributes
    ----------
    n_estimators_true : List[int]
        The true number of trees in each model (in case early stopping is used).
    """

    def __init__(
        self,
        n_repeats: int | None = 10,
        eval_percent: float = 0.1,
        n_jobs: int | None = -1,
        seed: int | None = None,
        score_parameterization: str | ScoreParameterization = "noise",
        noise_features: str | NoiseFeatureBuilder = "raw_time",
        **lgbm_args,
    ) -> None:
        self.n_repeats = n_repeats
        self.eval_percent = eval_percent
        self.n_jobs = n_jobs
        self.seed = seed
        self.score_parameterization = get_score_parameterization(score_parameterization)
        self.noise_feature_builder = get_noise_feature_builder(noise_features)

        self._lgbm_args = lgbm_args
        self.sde = None
        self.models = None  # Convention inputs are (y, x, t)
        self.n_estimators_true = None

    def score(
        self,
        y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Int[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        if self.sde is None:
            raise ValueError("The model has not been fitted yet.")
        assert self.models is not None

        scores = []
        predictors = self.noise_feature_builder.make_features(perturbed_y=y, X=X, t=t, sde=self.sde)
        _, std = self.sde.get_mean_std_pt_given_y0(y, t)
        for i in range(y.shape[-1]):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names.*",
                    category=UserWarning,
                )
                score_p = self.models[i].predict(predictors, num_threads=self.n_jobs)
            score = self.score_parameterization.reconstruct_score(
                prediction=score_p,
                perturbed_y=y[:, i],
                std=std[:, i],
                t=t,
            )
            scores.append(score)
        return np.array(scores).T

    def fit(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        sde: DiffusionSDE,
        cat_idx: list[int] | None = None,
    ):
        """
        Fit the score model to the data and the given SDE.

        Parameters
        ----------
        X : Float[np.ndarray, "batch x_dim"]
            The input data.
        y : Float[np.ndarray, "batch y_dim"]
            The true output values.
        sde : DiffusionSDE
            The SDE that the model is supposed to approximate the score of.
        cat_idx : Optional[List[int]]
            List of indices of categorical features in the input data. If `None`, all features are
            assumed to be continuous.
        """
        y_dim = y.shape[1]
        self.sde = sde

        lgb_X_train, lgb_X_val, lgb_y_train, lgb_y_val, cat_idx = _make_training_data(
            X=X,
            y=y,
            sde=sde,
            n_repeats=self.n_repeats,
            eval_percent=self.eval_percent,
            cat_idx=cat_idx,
            seed=self.seed,
            score_parameterization=self.score_parameterization,
            noise_feature_builder=self.noise_feature_builder,
        )

        models = []
        for i in range(y_dim):
            lgb_y_val_i = lgb_y_val[:, i] if lgb_y_val is not None else None
            score_model_i = _fit_one_lgbm_model(
                X=lgb_X_train,
                y=lgb_y_train[:, i],
                X_val=lgb_X_val,
                y_val=lgb_y_val_i,
                cat_idx=cat_idx,
                seed=self.seed,
                n_jobs=self.n_jobs,
                **self._lgbm_args,
            )
            models.append(score_model_i)
        self.models = models

        # collect the true number of trees learned by each model
        self.n_estimators_true = [model.n_estimators_ for model in self.models]
