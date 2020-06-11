import pickle

import gym
import pytest

from metarl.envs import MetaRLEnv
from tests.helpers import step_env_with_gym_quirks


class TestMetaRLEnv:

    def test_is_pickleable(self):
        env = MetaRLEnv(env_name='CartPole-v1')
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip.env.spec == env.env.spec

    @pytest.mark.nightly
    @pytest.mark.parametrize('spec', list(gym.envs.registry.all()))
    def test_all_gym_envs(self, spec):
        if spec._env_name.startswith('Defender'):
            pytest.skip(
                'Defender-* envs bundled in atari-py 0.2.x don\'t load')
        env = MetaRLEnv(spec.make())
        step_env_with_gym_quirks(env, spec)

    @pytest.mark.nightly
    @pytest.mark.parametrize('spec', list(gym.envs.registry.all()))
    def test_all_gym_envs_pickleable(self, spec):
        if spec._env_name.startswith('Defender'):
            pytest.skip(
                'Defender-* envs bundled in atari-py 0.2.x don\'t load')
        env = MetaRLEnv(env_name=spec.id)
        step_env_with_gym_quirks(env,
                                 spec,
                                 n=1,
                                 render=True,
                                 serialize_env=True)
