"""PyTorch algorithms."""
from metarl.torch.algos.ddpg import DDPG
# VPG has to been import first because it is depended by PPO and TRPO.
from metarl.torch.algos.vpg import VPG
from metarl.torch.algos.ppo import PPO  # noqa: I100
from metarl.torch.algos.trpo import TRPO
from metarl.torch.algos.maml_ppo import MAMLPPO  # noqa: I100
from metarl.torch.algos.maml_trpo import MAMLTRPO
from metarl.torch.algos.maml_vpg import MAMLVPG
from metarl.torch.algos.pearl import PEARL
from metarl.torch.algos.sac import SAC
from metarl.torch.algos.mtsac import MTSAC  # noqa: I100

__all__ = [
    'DDPG', 'VPG', 'PPO', 'TRPO', 'MAMLPPO', 'MAMLTRPO', 'MAMLVPG', 'MTSAC',
    'PEARL', 'SAC'
]
