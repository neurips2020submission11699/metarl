#!/usr/bin/env python3
"""An example to train a task with TRPO algorithm."""
import gym

from metarl import wrap_experiment
from metarl.envs import MetaRLEnv
from metarl.experiment import LocalTFRunner
from metarl.experiment.deterministic import set_seed
from metarl.np.baselines import LinearFeatureBaseline
from metarl.tf.algos import TRPO
from metarl.tf.policies import CategoricalMLPPolicy


@wrap_experiment
def trpo_gym_tf_cartpole(ctxt=None, seed=1):
    """Train TRPO with CartPole-v0 environment.

    Args:
        ctxt (metarl.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        env = MetaRLEnv(gym.make('CartPole-v0'))

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=200,
            discount=0.99,
            max_kl_step=0.01,
        )

        runner.setup(algo, env)
        runner.train(n_epochs=120, batch_size=4000)


trpo_gym_tf_cartpole()