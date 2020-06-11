#!/usr/bin/env python3
"""This is an example to train a task with CMA-ES.

Here it runs CartPole-v1 environment with 100 epoches.

Results:
    AverageReturn: 100
    RiseTime: epoch 38 (itr 760),
              but regression is observed in the course of training.
"""
from metarl import wrap_experiment
from metarl.envs import MetaRLEnv
from metarl.experiment import LocalTFRunner
from metarl.experiment.deterministic import set_seed
from metarl.np.algos import CMAES
from metarl.np.baselines import LinearFeatureBaseline
from metarl.sampler import OnPolicyVectorizedSampler
from metarl.tf.policies import CategoricalMLPPolicy


@wrap_experiment
def cma_es_cartpole(ctxt=None, seed=1):
    """Train CMA_ES with Cartpole-v1 environment.

    Args:
        ctxt (metarl.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(ctxt) as runner:
        env = MetaRLEnv(env_name='CartPole-v1')

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))
        baseline = LinearFeatureBaseline(env_spec=env.spec)

        n_samples = 20

        algo = CMAES(env_spec=env.spec,
                     policy=policy,
                     baseline=baseline,
                     max_path_length=100,
                     n_samples=n_samples)

        runner.setup(algo, env, sampler_cls=OnPolicyVectorizedSampler)
        runner.train(n_epochs=100, batch_size=1000)


cma_es_cartpole()
