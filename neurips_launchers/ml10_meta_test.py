"""This script is used for doing off-line meta testing."""
from metarl.envs.ml_wrapper import ML10WithPinnedGoal

from metarl.experiment.meta_test_helper import MetaTestHelper

if __name__ == "__main__":
    MetaTestHelper.read_cmd(ML10WithPinnedGoal.get_test_tasks)
