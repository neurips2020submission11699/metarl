"""PEARL benchmark script."""

import datetime
import os
import os.path as osp
import random

import akro
import dowel
from dowel import logger as dowel_logger
import numpy as np
import pytest
from metaworld.benchmarks import ML1

from metarl.envs import normalize
from metarl.envs.base import MetaRLEnv
from metarl.envs.env_spec import EnvSpec
from metarl.experiment import deterministic, LocalRunner
from metarl.experiment.task_sampler import SetTaskSampler
from metarl.experiment.snapshotter import SnapshotConfig
from metarl.sampler import PEARLSampler
from metarl.torch.algos import PEARLSAC
from metarl.torch.embeddings import MLPEncoder
from metarl.torch.q_functions import ContinuousMLPQFunction
from metarl.torch.policies import ContextConditionedPolicy, \
    TanhGaussianMLPPolicy2
import metarl.torch.utils as tu
from tests import benchmark_helper
import tests.helpers as Rh

# hyperparams for metarl
params = dict(
    num_epochs=1000,
    num_train_tasks=50,
    num_test_tasks=10,
    latent_size=7,
    net_size=300,
    meta_batch_size=16,
    num_steps_per_epoch=4000,
    num_initial_steps=4000,
    num_tasks_sample=15,
    num_steps_prior=750,
    num_extra_rl_steps_posterior=750,
    num_evals=5,
    num_steps_per_eval=450,
    batch_size=256,
    embedding_batch_size=64,
    embedding_mini_batch_size=64,
    max_path_length=150,
    reward_scale=10.,
    use_information_bottleneck=True,
    use_next_obs_in_context=False,
    n_trials=3,
    use_gpu=True,
)


class TestBenchmarkPEARL:
    '''Compare benchmarks between metarl and baselines.'''

    @pytest.mark.huge
    def test_benchmark_pearl(self):
        '''
        Compare benchmarks between metarl and baselines.
        :return:
        '''
        env_sampler = SetTaskSampler(lambda: MetaRLEnv(
            normalize(ML1.get_train_tasks('push-v1'))))
        env = env_sampler.sample(params['num_train_tasks'])
        test_env_sampler = SetTaskSampler(lambda: MetaRLEnv(
            normalize(ML1.get_test_tasks('push-v1'))))
        test_env = test_env_sampler.sample(params['num_train_tasks'])
        env_id = 'push-v1'
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = osp.join(os.getcwd(), 'data', 'local', 'benchmarks',
                                 'pearl', timestamp)
        result_json = {}
        seeds = random.sample(range(100), params['n_trials'])
        task_dir = osp.join(benchmark_dir, env_id)
        plt_file = osp.join(benchmark_dir, '{}_benchmark.png'.format(env_id))
        metarl_csvs = []

        for trial in range(params['n_trials']):
            seed = seeds[trial]
            trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
            metarl_dir = trial_dir + '/metarl'

            metarl_csv = run_metarl(env, test_env, seed, metarl_dir)
            metarl_csvs.append(metarl_csv)

        env.close()

        benchmark_helper.plot_average_over_trials(
            [metarl_csvs],
            ys=['Test/Average/SuccessRate'],
            plt_file=plt_file,
            env_id=env_id,
            x_label='TotalEnvSteps',
            y_label='Test/Average/SuccessRate',
            names=['metarl_pearl'],
        )

        factor_val = params['meta_batch_size'] * params['max_path_length']
        result_json[env_id] = benchmark_helper.create_json(
            [metarl_csvs],
            seeds=seeds,
            trials=params['n_trials'],
            xs=['TotalEnvSteps'],
            ys=['Test/Average/SuccessRate'],
            factors=[factor_val],
            names=['metarl_pearl'])

        Rh.write_file(result_json, 'PEARL')


def run_metarl(env, test_env, seed, log_dir):
    """Create metarl model and training."""

    deterministic.set_seed(seed)
    snapshot_config = SnapshotConfig(snapshot_dir=log_dir,
                                     snapshot_mode='gap',
                                     snapshot_gap=10)
    runner = LocalRunner(snapshot_config)

    obs_dim = int(np.prod(env[0]().observation_space.shape))
    action_dim = int(np.prod(env[0]().action_space.shape))
    reward_dim = 1

    # instantiate networks
    encoder_in_dim = obs_dim + action_dim + reward_dim
    encoder_out_dim = params['latent_size'] * 2
    net_size = params['net_size']

    context_encoder = MLPEncoder(input_dim=encoder_in_dim,
                                 output_dim=encoder_out_dim,
                                 hidden_sizes=[200, 200, 200])

    space_a = akro.Box(low=-1,
                       high=1,
                       shape=(obs_dim + params['latent_size'], ),
                       dtype=np.float32)
    space_b = akro.Box(low=-1, high=1, shape=(action_dim, ), dtype=np.float32)
    augmented_env = EnvSpec(space_a, space_b)

    qf1 = ContinuousMLPQFunction(env_spec=augmented_env,
                                 hidden_sizes=[net_size, net_size, net_size])

    qf2 = ContinuousMLPQFunction(env_spec=augmented_env,
                                 hidden_sizes=[net_size, net_size, net_size])

    obs_space = akro.Box(low=-1, high=1, shape=(obs_dim, ), dtype=np.float32)
    action_space = akro.Box(low=-1,
                            high=1,
                            shape=(params['latent_size'], ),
                            dtype=np.float32)
    vf_env = EnvSpec(obs_space, action_space)

    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    policy = TanhGaussianMLPPolicy2(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size])

    context_conditioned_policy = ContextConditionedPolicy(
        latent_dim=params['latent_size'],
        context_encoder=context_encoder,
        policy=policy,
        use_ib=params['use_information_bottleneck'],
        use_next_obs=params['use_next_obs_in_context'],
    )

    pearlsac = PEARLSAC(
        env=env,
        test_env=test_env,
        policy=context_conditioned_policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        num_train_tasks=params['num_train_tasks'],
        num_test_tasks=params['num_test_tasks'],
        latent_dim=params['latent_size'],
        meta_batch_size=params['meta_batch_size'],
        num_steps_per_epoch=params['num_steps_per_epoch'],
        num_initial_steps=params['num_initial_steps'],
        num_tasks_sample=params['num_tasks_sample'],
        num_steps_prior=params['num_steps_prior'],
        num_extra_rl_steps_posterior=params['num_extra_rl_steps_posterior'],
        num_evals=params['num_evals'],
        num_steps_per_eval=params['num_steps_per_eval'],
        batch_size=params['batch_size'],
        embedding_batch_size=params['embedding_batch_size'],
        embedding_mini_batch_size=params['embedding_mini_batch_size'],
        max_path_length=params['max_path_length'],
        reward_scale=params['reward_scale'],
    )

    tu.set_gpu_mode(params['use_gpu'], gpu_id=0)
    if params['use_gpu']:
        pearlsac.to()

    tabular_log_file = osp.join(log_dir, 'progress.csv')
    tensorboard_log_dir = osp.join(log_dir)
    dowel_logger.add_output(dowel.StdOutput())
    dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
    dowel_logger.add_output(dowel.TensorBoardOutput(tensorboard_log_dir))

    runner.setup(algo=pearlsac,
                 env=env,
                 sampler_cls=PEARLSampler,
                 sampler_args=dict(max_path_length=params['max_path_length']))
    runner.train(n_epochs=params['num_epochs'],
                 batch_size=params['batch_size'])

    dowel_logger.remove_all()

    return tabular_log_file


if __name__ == '__main__':
    test_cls = TestBenchmarkPEARL()
    test_cls.test_benchmark_pearl()
