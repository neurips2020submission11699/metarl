import gym
import pytest

from metarl.envs import MetaRLEnv
from metarl.experiment import LocalTFRunner
from metarl.np.baselines import LinearFeatureBaseline
from metarl.sampler import OnPolicyVectorizedSampler
from metarl.tf.algos import REPS
from metarl.tf.policies import CategoricalMLPPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase

configs = [(1, None, 4), (3, None, 12), (2, 3, 3)]


class TestOnPolicyVectorizedSampler(TfGraphTestCase):

    @pytest.mark.parametrize('cpus, n_envs, expected_n_envs', [*configs])
    def test_on_policy_vectorized_sampler_n_envs(self, cpus, n_envs,
                                                 expected_n_envs):
        with LocalTFRunner(snapshot_config, sess=self.sess,
                           max_cpus=cpus) as runner:
            env = MetaRLEnv(gym.make('CartPole-v0'))

            policy = CategoricalMLPPolicy(env_spec=env.spec,
                                          hidden_sizes=[32, 32])

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = REPS(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        max_path_length=100,
                        discount=0.99)

            runner.setup(algo, env, sampler_args=dict(n_envs=n_envs))

            assert isinstance(runner._sampler, OnPolicyVectorizedSampler)
            assert runner._sampler._n_envs == expected_n_envs

            env.close()
