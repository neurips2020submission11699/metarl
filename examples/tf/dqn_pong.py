#!/usr/bin/env python3
"""This is an example to train a task with DQN algorithm in pixel environment.

Here it creates a gym environment Pong, and trains a DQN with 1M steps.
"""
import click
import gym

from metarl import wrap_experiment
from metarl.envs import MetaRLEnv
from metarl.envs.wrappers.clip_reward import ClipReward
from metarl.envs.wrappers.episodic_life import EpisodicLife
from metarl.envs.wrappers.fire_reset import FireReset
from metarl.envs.wrappers.grayscale import Grayscale
from metarl.envs.wrappers.max_and_skip import MaxAndSkip
from metarl.envs.wrappers.noop import Noop
from metarl.envs.wrappers.resize import Resize
from metarl.envs.wrappers.stack_frames import StackFrames
from metarl.experiment import LocalTFRunner
from metarl.experiment.deterministic import set_seed
from metarl.np.exploration_policies import EpsilonGreedyPolicy
from metarl.replay_buffer import PathBuffer
from metarl.tf.algos import DQN
from metarl.tf.policies import DiscreteQfDerivedPolicy
from metarl.tf.q_functions import DiscreteCNNQFunction


@click.command()
@click.option('--buffer_size', type=int, default=int(5e4))
@click.option('--max_path_length', type=int, default=None)
@wrap_experiment
def dqn_pong(ctxt=None, seed=1, buffer_size=int(5e4), max_path_length=None):
    """Train DQN on PongNoFrameskip-v4 environment.

    Args:
        ctxt (metarl.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        buffer_size (int): Number of timesteps to store in replay buffer.
        max_path_length (int): Maximum length of a path after which a path is
            considered complete. This is used during testing to minimize the
            memory required to store a single path.

    """
    set_seed(seed)
    with LocalTFRunner(ctxt) as runner:
        n_epochs = 100
        steps_per_epoch = 20
        sampler_batch_size = 500
        num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size

        env = gym.make('PongNoFrameskip-v4')
        env = Noop(env, noop_max=30)
        env = MaxAndSkip(env, skip=4)
        env = EpisodicLife(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireReset(env)
        env = Grayscale(env)
        env = Resize(env, 84, 84)
        env = ClipReward(env)
        env = StackFrames(env, 4)

        env = MetaRLEnv(env, is_image=True)

        replay_buffer = PathBuffer(capacity_in_transitions=buffer_size)

        qf = DiscreteCNNQFunction(env_spec=env.spec,
                                  filters=(
                                              (32, (8, 8)),
                                              (64, (4, 4)),
                                              (64, (3, 3)),
                                          ),
                                  strides=(4, 2, 1),
                                  dueling=False)  # yapf: disable

        policy = DiscreteQfDerivedPolicy(env_spec=env.spec, qf=qf)
        exploration_policy = EpsilonGreedyPolicy(env_spec=env.spec,
                                                 policy=policy,
                                                 total_timesteps=num_timesteps,
                                                 max_epsilon=1.0,
                                                 min_epsilon=0.02,
                                                 decay_ratio=0.1)

        algo = DQN(env_spec=env.spec,
                   policy=policy,
                   qf=qf,
                   exploration_policy=exploration_policy,
                   replay_buffer=replay_buffer,
                   qf_lr=1e-4,
                   discount=0.99,
                   min_buffer_size=int(1e4),
                   max_path_length=max_path_length,
                   double_q=False,
                   n_train_steps=500,
                   steps_per_epoch=steps_per_epoch,
                   target_network_update_freq=2,
                   buffer_batch_size=32)

        runner.setup(algo, env)
        runner.train(n_epochs=n_epochs, batch_size=sampler_batch_size)


dqn_pong()
