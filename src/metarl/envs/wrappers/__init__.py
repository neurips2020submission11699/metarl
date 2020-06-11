"""gym.Env wrappers.

Used to transform an environment in a modular way.
It is also possible to apply multiple wrappers at the same
time.

Example:
    StackFrames(GrayScale(gym.make('env')))

"""
from metarl.envs.wrappers.atari_env import AtariEnv
from metarl.envs.wrappers.clip_reward import ClipReward
from metarl.envs.wrappers.episodic_life import EpisodicLife
from metarl.envs.wrappers.fire_reset import FireReset
from metarl.envs.wrappers.grayscale import Grayscale
from metarl.envs.wrappers.max_and_skip import MaxAndSkip
from metarl.envs.wrappers.noop import Noop
from metarl.envs.wrappers.resize import Resize
from metarl.envs.wrappers.stack_frames import StackFrames

__all__ = [
    'AtariEnv', 'ClipReward', 'EpisodicLife', 'FireReset', 'Grayscale',
    'MaxAndSkip', 'Noop', 'Resize', 'StackFrames'
]
