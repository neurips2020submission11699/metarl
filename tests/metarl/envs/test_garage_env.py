import pytest

from metarl.envs import EnvSpec, MetaRLEnv


class TestMetaRLEnv:

    def test_wraps_env_spec(self):
        metarl_env = MetaRLEnv(env_name='Pendulum-v0')
        assert isinstance(metarl_env.spec, EnvSpec)

    def test_closes_box2d(self):
        metarl_env = MetaRLEnv(env_name='CarRacing-v0')
        metarl_env.render()
        assert metarl_env.env.viewer is not None
        metarl_env.close()
        assert metarl_env.env.viewer is None

    @pytest.mark.mujoco
    def test_closes_mujoco(self):
        metarl_env = MetaRLEnv(env_name='Ant-v2')
        metarl_env.render()
        assert metarl_env.env.viewer is not None
        metarl_env.close()
        assert metarl_env.env.viewer is None

    def test_time_limit_env(self):
        metarl_env = MetaRLEnv(env_name='Pendulum-v0')
        metarl_env.reset()
        for _ in range(200):
            _, _, done, info = metarl_env.step(
                metarl_env.spec.action_space.sample())
        assert not done and info['TimeLimit.truncated']
        assert info['MetaRLEnv.TimeLimitTerminated']
