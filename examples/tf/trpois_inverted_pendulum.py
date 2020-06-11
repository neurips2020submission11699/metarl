#!/usr/bin/env python3
"""Example using TRPO with ISSampler.

Iterations alternate between live and importance sampled iterations.
"""
import gym

from metarl import wrap_experiment
from metarl.envs import MetaRLEnv, normalize
from metarl.experiment import LocalTFRunner
from metarl.experiment.deterministic import set_seed
from metarl.np.baselines import LinearFeatureBaseline
from metarl.sampler import ISSampler
from metarl.tf.algos import TRPO
from metarl.tf.policies import GaussianMLPPolicy


@wrap_experiment
def trpois_inverted_pendulum(ctxt=None, seed=1):
    """Train TRPO on InvertedPendulum-v2 with importance sampling.

    Args:
        ctxt (metarl.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(ctxt) as runner:
        env = MetaRLEnv(normalize(gym.make('InvertedPendulum-v2')))

        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=100,
                    discount=0.99,
                    max_kl_step=0.01)

        runner.setup(algo,
                     env,
                     sampler_cls=ISSampler,
                     sampler_args=dict(n_backtrack=1))
        runner.train(n_epochs=200, batch_size=4000)


trpois_inverted_pendulum()
