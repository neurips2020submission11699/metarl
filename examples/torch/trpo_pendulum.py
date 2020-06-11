#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm (PyTorch).

Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.
"""
import torch

from metarl import wrap_experiment
from metarl.envs import MetaRLEnv
from metarl.experiment import LocalRunner
from metarl.experiment.deterministic import set_seed
from metarl.torch.algos import TRPO
from metarl.torch.policies import GaussianMLPPolicy
from metarl.torch.value_functions import GaussianMLPValueFunction


@wrap_experiment
def trpo_pendulum(ctxt=None, seed=1):
    """Train TRPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (metarl.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    env = MetaRLEnv(env_name='InvertedDoublePendulum-v2')

    runner = LocalRunner(ctxt)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[32, 32],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    algo = TRPO(env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                max_path_length=100,
                discount=0.99,
                center_adv=False)

    runner.setup(algo, env)
    runner.train(n_epochs=100, batch_size=1024)


trpo_pendulum(seed=1)
