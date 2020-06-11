"""
Wrappers for the DeepMind Control Suite.

See https://github.com/deepmind/dm_control
"""
try:
    import dm_control  # noqa: F401
except ImportError:
    raise ImportError("To use metarl's dm_control wrappers, please install "
                      'metarl[dm_control].')

from metarl.envs.dm_control.dm_control_viewer import DmControlViewer
from metarl.envs.dm_control.dm_control_env import DmControlEnv  # noqa: I100

__all__ = ['DmControlViewer', 'DmControlEnv']
