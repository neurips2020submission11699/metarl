"""PyTorch-backed modules and algorithms."""
from metarl.torch._functions import compute_advantages
from metarl.torch._functions import dict_np_to_torch
from metarl.torch._functions import filter_valids
from metarl.torch._functions import flatten_batch
from metarl.torch._functions import global_device
from metarl.torch._functions import pad_to_last
from metarl.torch._functions import product_of_gaussians
from metarl.torch._functions import set_gpu_mode
from metarl.torch._functions import torch_to_np
from metarl.torch._functions import update_module_params

__all__ = [
    'compute_advantages', 'dict_np_to_torch', 'filter_valids', 'flatten_batch',
    'global_device', 'pad_to_last', 'product_of_gaussians', 'set_gpu_mode',
    'torch_to_np', 'update_module_params'
]
