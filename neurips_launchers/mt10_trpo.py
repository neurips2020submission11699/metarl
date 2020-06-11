#!/usr/bin/env python3
"""An example to train a task with TRPO algorithm."""
import gym

from metarl.experiment import run_experiment
from metarl.np.baselines import LinearFeatureBaseline
from metarl.tf.algos import TRPO
from metarl.tf.envs import TfEnv
from metarl.tf.experiment import LocalTFRunner
from metarl.tf.policies import GaussianMLPPolicy
import random
import tensorflow as tf
from metarl.envs import normalize
from metarl.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from metarl.experiment.deterministic import set_seed
from metarl.tf.algos import PPO
from metarl.tf.baselines import GaussianMLPBaseline
from metarl.tf.envs import TfEnv
from metarl.tf.experiment import LocalTFRunner
from metarl.tf.policies import GaussianMLPPolicy
from metaworld.envs.mujoco.env_dict import EASY_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import EASY_MODE_CLS_DICT
from metarl import wrap_experiment

MT10_envs_by_id = {
    task: env(*EASY_MODE_ARGS_KWARGS[task]['args'],
              **EASY_MODE_ARGS_KWARGS[task]['kwargs'])
    for (task, env) in EASY_MODE_CLS_DICT.items()
}

env_ids = ['reach-v1', 'push-v1', 'pick-place-v1', 'door-v1', 'drawer-open-v1', 'drawer-close-v1', 'button-press-topdown-v1', 'ped-insert-side-v1', 'window-open-v1', 'window-close-v1']
# env_ids = ['push-v1']
# env_ids = ['reach-v1']
# env_ids = ['pick-place-v1']

MT10_envs = [TfEnv(normalize(MT10_envs_by_id[i], normalize_reward=True)) for i in env_ids]


@wrap_experiment
def trpo_mt10(ctxt=None, seed=1):

    """Run task."""
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        env = MultiEnvWrapper(MT10_envs, env_ids, sample_strategy=round_robin_strategy)

        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(64, 64))

        # baseline = LinearFeatureBaseline(env_spec=env.spec)
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(64, 64),
                use_trust_region=False,
            ),
        )

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=150,
                    discount=0.99,
                    gae_lambda=0.97,
                    max_kl_step=0.01)

        runner.setup(algo, env)
        runner.train(n_epochs=1500, batch_size=len(MT10_envs)*10*150)


trpo_mt10(seed=1)
