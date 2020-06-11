"""Policies for TensorFlow-based algorithms."""
from metarl.tf.policies.categorical_cnn_policy import CategoricalCNNPolicy
from metarl.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
from metarl.tf.policies.categorical_lstm_policy import CategoricalLSTMPolicy
from metarl.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from metarl.tf.policies.continuous_mlp_policy import ContinuousMLPPolicy
from metarl.tf.policies.discrete_qf_derived_policy import (
    DiscreteQfDerivedPolicy)
from metarl.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from metarl.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from metarl.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from metarl.tf.policies.gaussian_mlp_task_embedding_policy import (
    GaussianMLPTaskEmbeddingPolicy)
from metarl.tf.policies.policy import Policy, StochasticPolicy
from metarl.tf.policies.task_embedding_policy import TaskEmbeddingPolicy

__all__ = [
    'Policy', 'StochasticPolicy', 'CategoricalCNNPolicy',
    'CategoricalGRUPolicy', 'CategoricalLSTMPolicy', 'CategoricalMLPPolicy',
    'ContinuousMLPPolicy', 'DiscreteQfDerivedPolicy', 'GaussianGRUPolicy',
    'GaussianLSTMPolicy', 'GaussianMLPPolicy',
    'GaussianMLPTaskEmbeddingPolicy', 'TaskEmbeddingPolicy'
]
