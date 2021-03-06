import gym
import pytest

from metarl.envs import MetaRLEnv, normalize
from metarl.experiment import LocalTFRunner
from metarl.np.baselines import LinearFeatureBaseline
from metarl.tf.algos import TNPG
from metarl.tf.policies import GaussianMLPPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase


class TestTNPG(TfGraphTestCase):

    @pytest.mark.mujoco_long
    def test_tnpg_inverted_pendulum(self):
        """Test TNPG with InvertedPendulum-v2 environment."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = MetaRLEnv(normalize(gym.make('InvertedPendulum-v2')))

            policy = GaussianMLPPolicy(name='policy',
                                       env_spec=env.spec,
                                       hidden_sizes=(32, 32))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TNPG(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        max_path_length=100,
                        discount=0.99,
                        optimizer_args=dict(reg_coeff=5e-1))

            runner.setup(algo, env)

            last_avg_ret = runner.train(n_epochs=10, batch_size=10000)
            assert last_avg_ret > 15

            env.close()
