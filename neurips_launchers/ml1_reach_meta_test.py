"""This script is used for doing off-line meta testing."""
from functools import partial

from metarl.envs.ml_wrapper import ML1WithPinnedGoal

from metarl.experiment.meta_test_helper import MetaTestHelper

if __name__ == "__main__":
    MetaTestHelper.read_cmd(
        partial(ML1WithPinnedGoal.get_test_tasks, 'reach-v1'))
