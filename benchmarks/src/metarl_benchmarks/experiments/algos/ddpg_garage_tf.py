"""A regression test for automatic benchmarking metarl-TensorFlow-DDPG."""
import gym
import tensorflow as tf

from metarl import wrap_experiment
from metarl.envs import MetaRLEnv, normalize
from metarl.experiment import deterministic
from metarl.experiment import LocalTFRunner
from metarl.np.exploration_policies import AddOrnsteinUhlenbeckNoise
from metarl.replay_buffer import PathBuffer
from metarl.tf.algos import DDPG
from metarl.tf.policies import ContinuousMLPPolicy
from metarl.tf.q_functions import ContinuousMLPQFunction

hyper_parameters = {
    'policy_lr': 1e-4,
    'qf_lr': 1e-3,
    'policy_hidden_sizes': [64, 64],
    'qf_hidden_sizes': [64, 64],
    'n_epochs': 500,
    'steps_per_epoch': 20,
    'n_rollout_steps': 100,
    'n_train_steps': 50,
    'discount': 0.9,
    'tau': 1e-2,
    'replay_buffer_size': int(1e6),
    'sigma': 0.2
}


@wrap_experiment
def ddpg_metarl_tf(ctxt, env_id, seed):
    """Create metarl TensorFlow DDPG model and training.

    Args:
        ctxt (metarl.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the
            snapshotter.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    with LocalTFRunner(ctxt) as runner:
        env = MetaRLEnv(normalize(gym.make(env_id)))

        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=hyper_parameters['policy_hidden_sizes'],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh)

        exploration_policy = AddOrnsteinUhlenbeckNoise(
            env.spec, policy, sigma=hyper_parameters['sigma'])

        qf = ContinuousMLPQFunction(
            env_spec=env.spec,
            hidden_sizes=hyper_parameters['qf_hidden_sizes'],
            hidden_nonlinearity=tf.nn.relu)

        replay_buffer = PathBuffer(
            capacity_in_transitions=hyper_parameters['replay_buffer_size'])

        algo = DDPG(env_spec=env.spec,
                    policy=policy,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    steps_per_epoch=hyper_parameters['steps_per_epoch'],
                    policy_lr=hyper_parameters['policy_lr'],
                    qf_lr=hyper_parameters['qf_lr'],
                    target_update_tau=hyper_parameters['tau'],
                    n_train_steps=hyper_parameters['n_train_steps'],
                    discount=hyper_parameters['discount'],
                    min_buffer_size=int(1e4),
                    exploration_policy=exploration_policy,
                    policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                    qf_optimizer=tf.compat.v1.train.AdamOptimizer)

        runner.setup(algo, env)
        runner.train(n_epochs=hyper_parameters['n_epochs'],
                     batch_size=hyper_parameters['n_rollout_steps'])
