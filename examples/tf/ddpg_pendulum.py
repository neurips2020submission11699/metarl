#!/usr/bin/env python3
"""This is an example to train a task with DDPG algorithm.

Here it creates a gym environment InvertedDoublePendulum. And uses a DDPG with
1M steps.

Results:
    AverageReturn: 250
    RiseTime: epoch 499
"""
import gym
import tensorflow as tf

from metarl import wrap_experiment
from metarl.envs import MetaRLEnv
from metarl.experiment import LocalTFRunner
from metarl.experiment.deterministic import set_seed
from metarl.np.exploration_policies import AddOrnsteinUhlenbeckNoise
from metarl.replay_buffer import PathBuffer
from metarl.tf.algos import DDPG
from metarl.tf.policies import ContinuousMLPPolicy
from metarl.tf.q_functions import ContinuousMLPQFunction


@wrap_experiment
def ddpg_pendulum(ctxt=None, seed=1):
    """Train DDPG with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (metarl.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        env = MetaRLEnv(gym.make('InvertedDoublePendulum-v2'))

        policy = ContinuousMLPPolicy(env_spec=env.spec,
                                     hidden_sizes=[64, 64],
                                     hidden_nonlinearity=tf.nn.relu,
                                     output_nonlinearity=tf.nn.tanh)

        exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec,
                                                       policy,
                                                       sigma=0.2)

        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=[64, 64],
                                    hidden_nonlinearity=tf.nn.relu)

        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

        ddpg = DDPG(env_spec=env.spec,
                    policy=policy,
                    policy_lr=1e-4,
                    qf_lr=1e-3,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    steps_per_epoch=20,
                    target_update_tau=1e-2,
                    n_train_steps=50,
                    discount=0.9,
                    min_buffer_size=int(1e4),
                    exploration_policy=exploration_policy,
                    policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                    qf_optimizer=tf.compat.v1.train.AdamOptimizer)

        runner.setup(algo=ddpg, env=env)

        runner.train(n_epochs=500, batch_size=100)


ddpg_pendulum(seed=1)
