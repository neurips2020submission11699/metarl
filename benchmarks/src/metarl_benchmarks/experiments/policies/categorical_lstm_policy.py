"""Benchmarking experiment of the CategoricalLSTMPolicy."""
import gym
import tensorflow as tf

from metarl import wrap_experiment
from metarl.envs import MetaRLEnv, normalize
from metarl.experiment import deterministic
from metarl.experiment import LocalTFRunner
from metarl.np.baselines import LinearFeatureBaseline
from metarl.tf.algos import PPO
from metarl.tf.policies import CategoricalLSTMPolicy


@wrap_experiment
def categorical_lstm_policy(ctxt, env_id, seed):
    """Create Categorical LSTM Policy on TF-PPO.

    Args:
        ctxt (metarl.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the
            snapshotter.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    with LocalTFRunner(ctxt, max_cpus=12) as runner:
        env = MetaRLEnv(normalize(gym.make(env_id)))

        policy = CategoricalLSTMPolicy(
            env_spec=env.spec,
            hidden_dim=32,
            hidden_nonlinearity=tf.nn.tanh,
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            policy_ent_coeff=0.0,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
                learning_rate=1e-3,
            ),
        )

        runner.setup(algo, env, sampler_args=dict(n_envs=12))
        runner.train(n_epochs=488, batch_size=2048)
