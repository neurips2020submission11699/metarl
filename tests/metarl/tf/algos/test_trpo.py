"""
This script creates a test that fails when metarl.tf.algos.TRPO performance is
too low.
"""
import gym
import pytest
import tensorflow as tf

from metarl.envs import MetaRLEnv, normalize
from metarl.experiment import deterministic
from metarl.experiment import LocalTFRunner
from metarl.experiment import snapshotter
from metarl.np.baselines import LinearFeatureBaseline
from metarl.tf.algos import TRPO
from metarl.tf.baselines import GaussianCNNBaseline
from metarl.tf.baselines import GaussianMLPBaseline
from metarl.tf.optimizers import FiniteDifferenceHvp
from metarl.tf.policies import CategoricalCNNPolicy
from metarl.tf.policies import CategoricalGRUPolicy
from metarl.tf.policies import CategoricalLSTMPolicy
from metarl.tf.policies import GaussianMLPPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase


class TestTRPO(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.env = MetaRLEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
        self.policy = GaussianMLPPolicy(
            env_spec=self.env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )
        self.baseline = GaussianMLPBaseline(
            env_spec=self.env.spec,
            regressor_args=dict(hidden_sizes=(32, 32)),
        )

    @pytest.mark.mujoco_long
    def test_trpo_pendulum(self):
        """Test TRPO with Pendulum environment."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = TRPO(env_spec=self.env.spec,
                        policy=self.policy,
                        baseline=self.baseline,
                        max_path_length=100,
                        discount=0.99,
                        gae_lambda=0.98,
                        policy_ent_coeff=0.0)
            runner.setup(algo, self.env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 40

    @pytest.mark.mujoco
    def test_trpo_unknown_kl_constraint(self):
        """Test TRPO with unkown KL constraints."""
        with pytest.raises(ValueError, match='Invalid kl_constraint'):
            TRPO(
                env_spec=self.env.spec,
                policy=self.policy,
                baseline=self.baseline,
                max_path_length=100,
                discount=0.99,
                gae_lambda=0.98,
                policy_ent_coeff=0.0,
                kl_constraint='random kl_constraint',
            )

    @pytest.mark.mujoco_long
    def test_trpo_soft_kl_constraint(self):
        """Test TRPO with unkown KL constraints."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = TRPO(env_spec=self.env.spec,
                        policy=self.policy,
                        baseline=self.baseline,
                        max_path_length=100,
                        discount=0.99,
                        gae_lambda=0.98,
                        policy_ent_coeff=0.0,
                        kl_constraint='soft')
            runner.setup(algo, self.env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 45

    @pytest.mark.mujoco_long
    def test_trpo_lstm_cartpole(self):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = MetaRLEnv(normalize(gym.make('CartPole-v1')))

            policy = CategoricalLSTMPolicy(name='policy', env_spec=env.spec)

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        max_path_length=100,
                        discount=0.99,
                        max_kl_step=0.01,
                        optimizer_args=dict(hvp_approach=FiniteDifferenceHvp(
                            base_eps=1e-5)))

            snapshotter.snapshot_dir = './'
            runner.setup(algo, env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 80

            env.close()

    @pytest.mark.mujoco_long
    def test_trpo_gru_cartpole(self):
        deterministic.set_seed(2)
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = MetaRLEnv(normalize(gym.make('CartPole-v1')))

            policy = CategoricalGRUPolicy(name='policy', env_spec=env.spec)

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        max_path_length=100,
                        discount=0.99,
                        max_kl_step=0.01,
                        optimizer_args=dict(hvp_approach=FiniteDifferenceHvp(
                            base_eps=1e-5)))

            runner.setup(algo, env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 80

            env.close()

    def teardown_method(self):
        self.env.close()
        super().teardown_method()


class TestTRPOCNNCubeCrash(TfGraphTestCase):

    @pytest.mark.large
    def test_trpo_cnn_cubecrash(self):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = MetaRLEnv(normalize(gym.make('CubeCrash-v0')))

            policy = CategoricalCNNPolicy(env_spec=env.spec,
                                          filters=((32, (8, 8)), (64, (4, 4))),
                                          strides=(4, 2),
                                          padding='VALID',
                                          hidden_sizes=(32, 32))

            baseline = GaussianCNNBaseline(
                env_spec=env.spec,
                regressor_args=dict(filters=((32, (8, 8)), (64, (4, 4))),
                                    strides=(4, 2),
                                    padding='VALID',
                                    hidden_sizes=(32, 32),
                                    use_trust_region=True))

            algo = TRPO(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        max_path_length=100,
                        discount=0.99,
                        gae_lambda=0.98,
                        max_kl_step=0.01,
                        policy_ent_coeff=0.0,
                        flatten_input=False)

            runner.setup(algo, env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > -1.5

            env.close()
