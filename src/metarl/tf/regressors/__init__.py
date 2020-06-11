"""Regressors for TensorFlow-based algorithms."""
from metarl.tf.regressors.bernoulli_mlp_regressor import BernoulliMLPRegressor
from metarl.tf.regressors.categorical_mlp_regressor import (
    CategoricalMLPRegressor)
from metarl.tf.regressors.continuous_mlp_regressor import (
    ContinuousMLPRegressor)
from metarl.tf.regressors.gaussian_cnn_regressor import GaussianCNNRegressor
from metarl.tf.regressors.gaussian_cnn_regressor_model import (
    GaussianCNNRegressorModel)
from metarl.tf.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from metarl.tf.regressors.regressor import Regressor, StochasticRegressor

__all__ = [
    'BernoulliMLPRegressor', 'CategoricalMLPRegressor',
    'ContinuousMLPRegressor', 'GaussianCNNRegressor',
    'GaussianCNNRegressorModel', 'GaussianMLPRegressor', 'Regressor',
    'StochasticRegressor'
]
