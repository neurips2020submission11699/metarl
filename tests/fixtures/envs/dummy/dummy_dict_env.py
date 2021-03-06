"""Dummy akro.Dict environment for testing purpose."""
import akro
import numpy as np

from metarl.envs import EnvSpec
from tests.fixtures.envs.dummy import DummyEnv


class DummyDictEnv(DummyEnv):
    """A dummy akro.Dict environment with predefined inner spaces.

    Args:
        random (bool): If observations are randomly generated or not.

    """

    def __init__(self, random=True):
        super().__init__(random)
        self.spec = EnvSpec(action_space=self.action_space,
                            observation_space=self.observation_space)

    @property
    def observation_space(self):
        """Return the observation space.

        Returns:
            akro.Dict: Observation space.

        """

        return akro.Dict({
            'achieved_goal':
            akro.Box(low=-200., high=200., shape=(3, ), dtype=np.float32),
            'desired_goal':
            akro.Box(low=-200., high=200., shape=(3, ), dtype=np.float32),
            'observation':
            akro.Box(low=-200., high=200., shape=(25, ), dtype=np.float32)
        })

    @property
    def action_space(self):
        """Return the action space.

        Returns:
            akro.Box: Action space.

        """
        return akro.Box(low=-5.0, high=5.0, shape=(1, ), dtype=np.float32)

    def reset(self):
        """Reset the environment.

        Returns:
            numpy.ndarray: Observation after reset.

        """
        return self.observation_space.sample()

    def step(self, action):
        """Step the environment.

        Args:
            action (int): Action input.

        Returns:
            np.ndarray: Observation.
            float: Reward.
            bool: If the environment is terminated.
            dict: Environment information.

        """
        return self.observation_space.sample(), 0, True, dict()

    # pylint: disable=no-self-use
    def compute_reward(self, achieved_goal, goal, info):
        """Function to compute new reward.

        Args:
            achieved_goal (numpy.ndarray): Achieved goal.
            goal (numpy.ndarray): Original desired goal.
            info (dict): Extra information.

        Returns:
            float: New computed reward.

        """
        del info
        return np.sum(achieved_goal - goal)
