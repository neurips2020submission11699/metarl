"""Tensorflow implementation of reinforcement learning algorithms."""
from metarl.tf.algos.ddpg import DDPG
from metarl.tf.algos.dqn import DQN
from metarl.tf.algos.erwr import ERWR
from metarl.tf.algos.npo import NPO
from metarl.tf.algos.ppo import PPO
from metarl.tf.algos.reps import REPS
from metarl.tf.algos.rl2 import RL2
from metarl.tf.algos.rl2ppo import RL2PPO
from metarl.tf.algos.rl2trpo import RL2TRPO
from metarl.tf.algos.td3 import TD3
from metarl.tf.algos.te_npo import TENPO
from metarl.tf.algos.te_ppo import TEPPO
from metarl.tf.algos.tnpg import TNPG
from metarl.tf.algos.trpo import TRPO
from metarl.tf.algos.vpg import VPG

__all__ = [
    'DDPG',
    'DQN',
    'ERWR',
    'NPO',
    'PPO',
    'REPS',
    'RL2',
    'RL2PPO',
    'RL2TRPO',
    'TD3',
    'TNPG',
    'TRPO',
    'VPG',
    'TENPO',
    'TEPPO',
]
