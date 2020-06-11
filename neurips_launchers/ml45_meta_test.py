"""This script is used for doing off-line meta testing."""
from metaworld.benchmarks import ML45WithPinnedGoal

from metarl.experiment.meta_test_helper import MetaTestHelper

if __name__ == "__main__":
    MetaTestHelper.read_cmd(ML45WithPinnedGoal.get_test_tasks)
