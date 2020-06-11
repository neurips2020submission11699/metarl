#!/usr/bin/env python3
"""This is an example to train a task with REPS algorithm.

Here it runs gym CartPole env with 100 iterations.

Results:
    AverageReturn: 100 +/- 40
    RiseTime: itr 10 +/- 5

"""

import gym

from metarl import wrap_experiment
from metarl.envs import MetaRLEnv
from metarl.experiment import LocalTFRunner
from metarl.experiment.deterministic import set_seed
from metarl.np.baselines import LinearFeatureBaseline
from metarl.tf.algos import REPS
from metarl.tf.policies import CategoricalMLPPolicy


@wrap_experiment
def reps_gym_cartpole(ctxt=None, seed=1):
    """Train REPS with CartPole-v0 environment.

    Args:
        ctxt (metarl.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        env = MetaRLEnv(gym.make('CartPole-v0'))

        policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=[32, 32])

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = REPS(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=100,
                    discount=0.99)

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=4000, plot=False)


reps_gym_cartpole()
