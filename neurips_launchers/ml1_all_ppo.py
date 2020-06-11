#!/usr/bin/env python3
"""
This is an example to train a task with PPO algorithm.

Here it creates InvertedDoublePendulum using gym. And uses a PPO with 1M
steps.

Results:
    AverageDiscountedReturn: 500
    RiseTime: itr 40

"""
import sys

import numpy as np
import tensorflow as tf
from metarl import wrap_experiment
from metarl.envs import MetaRLEnv, normalize_reward
from metarl.envs.ml_wrapper import ML1WithPinnedGoal
from metarl.envs.multi_task_metaworld_wrapper import MTMetaWorldWrapper
from metarl.experiment.deterministic import set_seed
from metarl.tf.algos import PPO
from metarl.tf.baselines import GaussianMLPBaseline
from metarl.tf.experiment import LocalTFRunner
from metarl.tf.policies import GaussianMLPPolicy


# env_id = 'push-v1'
# env_id = 'reach-v1'
# env_id = 'pick-place-v1'

@wrap_experiment
def ppo_ml1(ctxt=None, seed=1):

    """Run task."""
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        Ml1_reach_envs = get_ML1_envs_test(env_id)
        env = MTMetaWorldWrapper(Ml1_reach_envs)

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
            hidden_w_init=tf.constant_initializer(np.sqrt(2)),
            hidden_b_init=tf.constant_initializer(np.sqrt(2)),
        )

        # baseline = LinearFeatureBaseline(env_spec=env.spec)
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(64, 64),
                use_trust_region=False,
                hidden_w_init=tf.constant_initializer(np.sqrt(2)),
                hidden_b_init=tf.constant_initializer(np.sqrt(2)),
            ),
        )

        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=150,
            discount=0.99,
            gae_lambda=0.97,
            lr_clip_range=0.2,
            optimizer_args=dict(
                batch_size=30,
                max_epochs=4,
                tf_optimizer_args=dict(
                    learning_rate=3e-4,
                ),
            ),
        )

        timesteps = 6000000
        batch_size = 150 * env.num_tasks
        epochs = timesteps // batch_size

        print (f'epochs: {epochs}, batch_size: {batch_size}')

        runner.setup(algo, env, sampler_args={'n_envs': 1})
        runner.train(n_epochs=epochs, batch_size=batch_size, plot=False)


def get_ML1_envs_test(name):
    bench = ML1WithPinnedGoal.get_train_tasks(name)
    tasks = [{'task': 0, 'goal': i} for i in range(50)]
    ret = {}
    for task in tasks:
        new_bench = bench.clone(bench)
        new_bench.set_task(task)
        ret[(env_id+"_"+str(task['goal']))] = MetaRLEnv(normalize_reward(new_bench.active_env))
    return ret


# env_id = 'push-v1'
# env_id = 'reach-v1'
# env_id = 'pick-place-v1'

assert len(sys.argv) > 1

env_id = sys.argv[1]
s = int(sys.argv[2]) if len(sys.argv) > 2 else 0

ppo_ml1(seed=s)
