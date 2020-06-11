"""A regression test for automatic benchmarking metarl-PyTorch-PPO."""
import gym
import torch

from metarl import wrap_experiment
from metarl.envs import MetaRLEnv, normalize
from metarl.experiment import deterministic, LocalRunner
from metarl.torch.algos import PPO as PyTorch_PPO
from metarl.torch.optimizers import OptimizerWrapper
from metarl.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from metarl.torch.value_functions import GaussianMLPValueFunction

hyper_parameters = {
    'n_epochs': 500,
    'max_path_length': 100,
    'batch_size': 1024,
}


@wrap_experiment
def ppo_metarl_pytorch(ctxt, env_id, seed):
    """Create metarl PyTorch PPO model and training.

    Args:
        ctxt (metarl.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the
            snapshotter.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    runner = LocalRunner(ctxt)

    env = MetaRLEnv(normalize(gym.make(env_id)))

    policy = PyTorch_GMP(env.spec,
                         hidden_sizes=(32, 32),
                         hidden_nonlinearity=torch.tanh,
                         output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    policy_optimizer = OptimizerWrapper((torch.optim.Adam, dict(lr=2.5e-4)),
                                        policy,
                                        max_optimization_epochs=10,
                                        minibatch_size=64)

    vf_optimizer = OptimizerWrapper((torch.optim.Adam, dict(lr=2.5e-4)),
                                    value_function,
                                    max_optimization_epochs=10,
                                    minibatch_size=64)

    algo = PyTorch_PPO(env_spec=env.spec,
                       policy=policy,
                       value_function=value_function,
                       policy_optimizer=policy_optimizer,
                       vf_optimizer=vf_optimizer,
                       max_path_length=hyper_parameters['max_path_length'],
                       discount=0.99,
                       gae_lambda=0.95,
                       center_adv=True,
                       lr_clip_range=0.2)

    runner.setup(algo, env)
    runner.train(n_epochs=hyper_parameters['n_epochs'],
                 batch_size=hyper_parameters['batch_size'])