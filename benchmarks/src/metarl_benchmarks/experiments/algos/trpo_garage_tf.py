"""A regression test for automatic benchmarking metarl-TensorFlow-TRPO."""
import gym
import tensorflow as tf

from metarl import wrap_experiment
from metarl.envs import MetaRLEnv, normalize
from metarl.experiment import deterministic
from metarl.experiment import LocalTFRunner
from metarl.np.baselines import LinearFeatureBaseline
from metarl.tf.algos import TRPO
from metarl.tf.policies import GaussianMLPPolicy

hyper_parameters = {
    'hidden_sizes': [32, 32],
    'max_kl': 0.01,
    'gae_lambda': 0.97,
    'discount': 0.99,
    'max_path_length': 100,
    'n_epochs': 999,
    'batch_size': 1024,
}


@wrap_experiment
def trpo_metarl_tf(ctxt, env_id, seed):
    """Create metarl Tensorflow TROI model and training.

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

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=hyper_parameters['hidden_sizes'],
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=hyper_parameters['max_path_length'],
                    discount=hyper_parameters['discount'],
                    gae_lambda=hyper_parameters['gae_lambda'],
                    max_kl_step=hyper_parameters['max_kl'])

        runner.setup(algo, env)
        runner.train(n_epochs=hyper_parameters['n_epochs'],
                     batch_size=hyper_parameters['batch_size'])
