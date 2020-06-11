"""This script creates a regression test over metarl-MAML and ProMP-TRPO.

Unlike metarl, baselines doesn't set max_path_length. It keeps steps the action
until it's done. So we introduced tests.wrappers.AutoStopEnv wrapper to set
done=True when it reaches max_path_length. We also need to change the
metarl.tf.samplers.BatchSampler to smooth the reward curve.
"""
import argparse
import datetime
import os
import os.path as osp
import random
import sys

import numpy as np
import dowel
from dowel import logger as dowel_logger
import pytest
import torch
import tensorflow as tf

from metarl.envs.ml_wrapper import ML1WithPinnedGoal

from metarl.envs import normalize
from metarl.envs.base import MetaRLEnv
from metarl.envs import TaskIdWrapper2
from metarl.experiment import deterministic, LocalRunner, SnapshotConfig
from metarl.np.baselines import LinearFeatureBaseline
from metarl.torch.algos import MAMLTRPO
from metarl.torch.policies import GaussianMLPPolicy

from tests import benchmark_helper
import tests.helpers as Rh

test_metarl = True

hyper_parameters = {
    'hidden_sizes': [100, 100],
    'max_kl': 0.01,
    'inner_lr': 0.05,
    'gae_lambda': 1.0,
    'discount': 0.99,
    'max_path_length': 100,
    'fast_batch_size': 10,  # num of rollouts per task
    'meta_batch_size': 20,  # num of tasks
    'n_epochs': 2500,
    # 'n_epochs': 1,
    'n_trials': 3,
    'num_grad_update': 1,
    'n_parallel': 1,
    'inner_loss': 'log_likelihood'
}


class TestBenchmarkMAML:  # pylint: disable=too-few-public-methods
    """Compare benchmarks between metarl and baselines."""

    @pytest.mark.huge
    def test_benchmark_maml(self, _):  # pylint: disable=no-self-use
        """Compare benchmarks between metarl and baselines."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = './data/local/benchmarks/maml-ml1-push/%s/' % timestamp
        result_json = {}
        env_id = 'ML1-Push'
        meta_env = TaskIdWrapper2(ML1WithPinnedGoal.get_train_tasks('push-v1'))

        seeds = random.sample(range(100), hyper_parameters['n_trials'])
        task_dir = osp.join(benchmark_dir, env_id)
        plt_file = osp.join(benchmark_dir, '{}_benchmark.png'.format(env_id))
        promp_csvs = []
        metarl_csvs = []

        for trial in range(hyper_parameters['n_trials']):
            seed = seeds[trial]
            trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
            metarl_dir = trial_dir + '/metarl'
            promp_dir = trial_dir + '/promp'

            if test_metarl:
                # Run metarl algorithm
                env = MetaRLEnv(normalize(meta_env, expected_action_scale=10.))
                metarl_csv = run_metarl(env, seed, metarl_dir)
                metarl_csvs.append(metarl_csv)
                env.close()

def run_metarl(env, seed, log_dir):
    """Create metarl PyTorch MAML model and training.

    Args:
        env (MetaRLEnv): Environment of the task.
        seed (int): Random positive integer for the trial.
        log_dir (str): Log dir path.

    Returns:
        str: Path to output csv file

    """
    deterministic.set_seed(seed)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=hyper_parameters['hidden_sizes'],
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = MAMLTRPO(env=env,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=hyper_parameters['max_path_length'],
                    discount=hyper_parameters['discount'],
                    gae_lambda=hyper_parameters['gae_lambda'],
                    meta_batch_size=hyper_parameters['meta_batch_size'],
                    inner_lr=hyper_parameters['inner_lr'],
                    max_kl_step=hyper_parameters['max_kl'],
                    num_grad_updates=hyper_parameters['num_grad_update'])

    # Set up logger since we are not using run_experiment
    tabular_log_file = osp.join(log_dir, 'progress.csv')
    dowel_logger.add_output(dowel.StdOutput())
    dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
    dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

    snapshot_config = SnapshotConfig(snapshot_dir=log_dir,
                                     snapshot_mode='all',
                                     snapshot_gap=1)

    runner = LocalRunner(snapshot_config=snapshot_config)
    runner.setup(algo, env, sampler_args=dict(n_envs=5))
    runner.train(n_epochs=hyper_parameters['n_epochs'],
                 batch_size=(hyper_parameters['fast_batch_size'] *
                             hyper_parameters['max_path_length']))

    dowel_logger.remove_all()

    return tabular_log_file


def worker(variant):
    variant_str = '-'.join(['{}_{}'.format(k, v) for k, v in variant.items()])
    if 'hidden_sizes' in variant:
        hidden_sizes = variant['hidden_sizes']
        variant['hidden_sizes'] = [hidden_sizes, hidden_sizes]
    hyper_parameters.update(variant)

    test_cls = TestBenchmarkMAML()
    test_cls.test_benchmark_maml(variant_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--combined', action='store_true', default=False)

    known_args, unknown_args = parser.parse_known_args()

    for arg in unknown_args:
        if arg.startswith('--'):
            parser.add_argument(arg, type=float)

    args = parser.parse_args()
    print(args)

    test_metarl = True

    parallel = args.parallel
    combined = args.combined
    args = vars(args)
    del args['parallel']
    del args['combined']

    n_variants = len(args)
    if combined:
        variants = [{
            k: int(v) if v.is_integer() else v for k, v in args.items()
        }]
    else:
        if n_variants > 0:
            variants = [{
                k: int(v) if v.is_integer() else v
            } for k, v in args.items()]
        else:
            variants = [dict(n_trials=1)
                        for _ in range(hyper_parameters['n_trials'])]

    for key in args:
        assert key in hyper_parameters, "{} is not a hyperparameter".format(key)

    children = []
    for i, variant in enumerate(variants):
        random.seed(i)
        pid = os.fork()
        if pid == 0:
            worker(variant)
            exit()
        else:
            if parallel:
                children.append(pid)
            else:
                os.waitpid(pid, 0)

    if parallel:
        for child in children:
            os.waitpid(child, 0)
