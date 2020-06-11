"""Baselines (value functions) which use NumPy as a numerical backend."""
from metarl.np.baselines.baseline import Baseline
from metarl.np.baselines.linear_feature_baseline import LinearFeatureBaseline
from metarl.np.baselines.linear_multi_feature_baseline import (
    LinearMultiFeatureBaseline)
from metarl.np.baselines.zero_baseline import ZeroBaseline

__all__ = [
    'Baseline', 'LinearFeatureBaseline', 'LinearMultiFeatureBaseline',
    'ZeroBaseline'
]
