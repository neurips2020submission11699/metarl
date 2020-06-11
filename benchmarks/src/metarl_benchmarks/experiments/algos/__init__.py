"""Benchmarking experiments for algorithms."""
from metarl_benchmarks.experiments.algos.ddpg_metarl_tf import ddpg_metarl_tf
from metarl_benchmarks.experiments.algos.her_metarl_tf import her_metarl_tf
from metarl_benchmarks.experiments.algos.ppo_metarl_pytorch import (
    ppo_metarl_pytorch)
from metarl_benchmarks.experiments.algos.ppo_metarl_tf import ppo_metarl_tf
from metarl_benchmarks.experiments.algos.td3_metarl_tf import td3_metarl_tf
from metarl_benchmarks.experiments.algos.trpo_metarl_pytorch import (
    trpo_metarl_pytorch)
from metarl_benchmarks.experiments.algos.trpo_metarl_tf import trpo_metarl_tf
from metarl_benchmarks.experiments.algos.vpg_metarl_pytorch import (
    vpg_metarl_pytorch)
from metarl_benchmarks.experiments.algos.vpg_metarl_tf import vpg_metarl_tf

__all__ = [
    'ddpg_metarl_tf', 'her_metarl_tf', 'ppo_metarl_pytorch', 'ppo_metarl_tf',
    'td3_metarl_tf', 'trpo_metarl_pytorch', 'trpo_metarl_tf',
    'vpg_metarl_pytorch', 'vpg_metarl_tf'
]
