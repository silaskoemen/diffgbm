from __future__ import annotations

import warnings
from typing import Literal
from typing import cast

import numpy as np
from jaxtyping import Float
from numpy import ndarray
from sklearn.model_selection import KFold

from treeffuser._score_models import _fit_one_lgbm_model

ResidualizeMode = Literal["off", "mean", "mean_scale"]


class ConditionalResidualizer:
    """
    Cross-fitted conditional mean and scale residualizer.

    The residualizer is fitted on already-preprocessed `X` and globally standardized `y`.
    It trains the diffusion on out-of-fold residuals, then uses averaged fold predictions
    for future transforms and sample inversion. On training rows, these two residual
    transforms are not identical because most fold models saw each row. The gap is
    controlled by the residualizer model capacity and is covered by tests.

    OOF residuals are centered before global residual scaling. The center is stored and
    re-added during inverse transforms, so the residualizer remains an invertible affine
    transform for fixed model predictions. The global residual scale uses a 1%/99%
    winsorized standard deviation as its primary estimator, with MAD and then 1.0 as
    degenerate fallbacks. This keeps the diffusion target close to unit variance while
    limiting single-row leverage.
    """

    def __init__(
        self,
        residualize: ResidualizeMode,
        k_folds: int = 5,
        seed: int | None = None,
        extra_params: dict | None = None,
    ) -> None:
        if residualize not in ("mean", "mean_scale"):
            raise ValueError("ConditionalResidualizer requires residualize to be 'mean' or 'mean_scale'.")
        if k_folds < 2:
            raise ValueError("k_folds must be at least 2.")

        self.residualize = residualize
        self.k_folds = k_folds
        self.seed = seed
        self.extra_params = extra_params or {}

        self.mean_models: list[list] | None = None
        self.scale_models: list[list] | None = None
        self.residual_center: Float[ndarray, "1 y_dim"] | None = None
        self.residual_global_scale: Float[ndarray, "1 y_dim"] | None = None
        self.scale_floor: Float[ndarray, "1 y_dim"] | None = None
        self.eps: Float[ndarray, "1 y_dim"] | None = None
        self.mean_oof: Float[ndarray, "batch y_dim"] | None = None
        self.scale_oof: Float[ndarray, "batch y_dim"] | None = None
        self.effective_k_folds: int | None = None
        self._is_fitted = False

    def fit_transform(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        cat_idx: list[int] | None = None,
    ) -> Float[ndarray, "batch y_dim"]:
        self.fit(X=X, y=y, cat_idx=cat_idx)
        return self._training_residuals(y)

    def fit(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        cat_idx: list[int] | None = None,
    ) -> "ConditionalResidualizer":
        n, y_dim = y.shape
        effective_k = min(self.k_folds, max(2, n // 40))
        if effective_k < 2 or n < 80:
            raise ValueError("Residualization requires at least 80 training rows.")
        if effective_k < self.k_folds:
            warnings.warn(
                f"Reducing residualize_k_folds from {self.k_folds} to {effective_k} " f"for {n} training rows.",
                UserWarning,
                stacklevel=2,
            )

        self.effective_k_folds = effective_k
        splits = list(KFold(n_splits=effective_k, shuffle=True, random_state=self.seed).split(X))

        self.mean_models = [[] for _ in range(y_dim)]
        mean_oof = np.empty_like(y)
        for dim in range(y_dim):
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                model = self._fit_model(
                    X=X[train_idx],
                    y=y[train_idx, dim],
                    cat_idx=cat_idx,
                    seed=self._model_seed(kind_offset=10_000, dim=dim, fold_idx=fold_idx),
                )
                self.mean_models[dim].append(model)
                mean_oof[val_idx, dim] = self._predict_model(model, X[val_idx])

        self.mean_oof = mean_oof
        residual_oof = y - mean_oof
        raw_residual_oof = residual_oof

        if self.residualize == "mean_scale":
            scale_oof = self._fit_scale_models(
                X=X,
                y=y,
                residual_oof=residual_oof,
                splits=splits,
                cat_idx=cat_idx,
            )
            self.scale_oof = scale_oof
            raw_residual_oof = residual_oof / scale_oof

        self.residual_center = np.mean(raw_residual_oof, axis=0, keepdims=True)
        centered_residual_oof = raw_residual_oof - self.residual_center
        self.residual_global_scale = self._robust_scale(centered_residual_oof).reshape(1, -1)
        self._is_fitted = True
        return self

    def transform(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
    ) -> Float[ndarray, "batch y_dim"]:
        self._check_is_fitted()
        assert self.residual_center is not None
        assert self.residual_global_scale is not None

        mean = self.predict_mean(X)
        residual = y - mean
        if self.residualize == "mean_scale":
            residual = residual / self.predict_scale(X)
        return (residual - self.residual_center) / self.residual_global_scale

    def inverse_transform(
        self,
        X: Float[ndarray, "batch x_dim"],
        residual: Float[ndarray, "batch y_dim"],
    ) -> Float[ndarray, "batch y_dim"]:
        self._check_is_fitted()
        assert self.residual_center is not None
        assert self.residual_global_scale is not None

        mean = self.predict_mean(X)
        scaled_residual = residual * self.residual_global_scale + self.residual_center
        if self.residualize == "mean_scale":
            scaled_residual = scaled_residual * self.predict_scale(X)
        return mean + scaled_residual

    def predict_mean(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        self._check_is_fitted()
        assert self.mean_models is not None
        return np.column_stack([self._predict_model_average(models, X) for models in self.mean_models])

    def predict_scale(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch y_dim"]:
        self._check_is_fitted()
        if self.residualize != "mean_scale":
            return np.ones((X.shape[0], self._y_dim()))
        assert self.scale_models is not None
        assert self.scale_floor is not None
        log_scale = np.column_stack([self._predict_model_average(models, X) for models in self.scale_models])
        return np.maximum(np.exp(log_scale), self.scale_floor)

    def _fit_scale_models(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch y_dim"],
        residual_oof: Float[ndarray, "batch y_dim"],
        splits: list[tuple[np.ndarray, np.ndarray]],
        cat_idx: list[int] | None,
    ) -> Float[ndarray, "batch y_dim"]:
        y_dim = residual_oof.shape[1]
        abs_residual = np.abs(residual_oof)
        eps = np.maximum(0.01 * np.median(abs_residual, axis=0, keepdims=True), np.finfo(float).eps)
        scale_target = np.log(abs_residual + eps)

        self.scale_models = [[] for _ in range(y_dim)]
        log_scale_oof = np.empty_like(residual_oof)
        for dim in range(y_dim):
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                model = self._fit_model(
                    X=X[train_idx],
                    y=scale_target[train_idx, dim],
                    cat_idx=cat_idx,
                    seed=self._model_seed(kind_offset=20_000, dim=dim, fold_idx=fold_idx),
                )
                self.scale_models[dim].append(model)
                log_scale_oof[val_idx, dim] = self._predict_model(model, X[val_idx])

        scale_raw_oof = np.exp(log_scale_oof)
        y_std = np.std(y, axis=0, keepdims=True)
        floor_from_scale = 0.05 * np.median(scale_raw_oof, axis=0, keepdims=True)
        floor_from_y = 0.01 * y_std
        self.scale_floor = np.maximum(floor_from_scale, floor_from_y)
        self.eps = eps
        return np.maximum(scale_raw_oof, self.scale_floor)

    def _training_residuals(self, y: Float[ndarray, "batch y_dim"]) -> Float[ndarray, "batch y_dim"]:
        assert self.mean_oof is not None
        assert self.residual_center is not None
        assert self.residual_global_scale is not None
        residual = y - self.mean_oof
        if self.residualize == "mean_scale":
            assert self.scale_oof is not None
            residual = residual / self.scale_oof
        return (residual - self.residual_center) / self.residual_global_scale

    def _fit_model(
        self,
        X: Float[ndarray, "batch x_dim"],
        y: Float[ndarray, "batch"],
        cat_idx: list[int] | None,
        seed: int | None,
    ):
        params = self._model_params()
        verbose = cast(int, params.pop("verbose"))
        n_jobs = cast(int, params.pop("n_jobs"))
        early_stopping_rounds = cast(int | None, params.pop("early_stopping_rounds"))
        return _fit_one_lgbm_model(
            X=X,
            y=y,
            X_val=None,
            y_val=None,
            seed=seed,
            verbose=verbose,
            cat_idx=cat_idx,
            n_jobs=n_jobs,
            early_stopping_rounds=early_stopping_rounds,
            **params,
        )

    def _model_params(self) -> dict:
        params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 1.0,
            "subsample_freq": 0,
            "early_stopping_rounds": None,
            "verbose": -1,
            "n_jobs": -1,
        }
        params.update(self.extra_params)
        return params

    def _model_seed(self, kind_offset: int, dim: int, fold_idx: int) -> int | None:
        if self.seed is None:
            return None
        return self.seed + kind_offset + 1_000 * dim + fold_idx

    @staticmethod
    def _robust_scale(residual: Float[ndarray, "batch y_dim"]) -> Float[ndarray, "y_dim"]:
        # Prefer winsorized std over MAD here because the downstream SDE and EDM defaults
        # expect approximately unit-variance residuals, not just a robust central scale.
        scale = _winsorized_std(residual)
        degenerate = scale <= np.finfo(float).eps
        if np.any(degenerate):
            median = np.median(residual[:, degenerate], axis=0)
            mad = np.median(np.abs(residual[:, degenerate] - median), axis=0)
            scale[degenerate] = 1.4826 * mad
        scale = np.where(scale <= np.finfo(float).eps, 1.0, scale)
        return scale

    @staticmethod
    def _predict_model_average(models: list, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch"]:
        predictions = [ConditionalResidualizer._predict_model(model, X) for model in models]
        return np.mean(predictions, axis=0)

    @staticmethod
    def _predict_model(model, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch"]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            return model.predict(X)

    def _y_dim(self) -> int:
        assert self.mean_models is not None
        return len(self.mean_models)

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError("The residualizer has not been fitted yet.")


def _winsorized_std(residual: Float[ndarray, "batch y_dim"]) -> Float[ndarray, "y_dim"]:
    low = np.quantile(residual, 0.01, axis=0)
    high = np.quantile(residual, 0.99, axis=0)
    clipped = np.clip(residual, low, high)
    return np.std(clipped, axis=0)
