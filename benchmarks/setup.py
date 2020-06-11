"""Setup script for metarl benchmarking scripts.

This package is generally not needed by users of metarl.
"""
import os

from setuptools import find_packages, setup

METARL_GH_TOKEN = os.environ.get('METARL_GH_TOKEN') or 'git'

REQUIRED = [
    # Please keep alphabetized
    'baselines @ https://{}@api.github.com/repos/openai/baselines/tarball/ea25b9e8b234e6ee1bca43083f8f3cf974143998'.format(METARL_GH_TOKEN),  # noqa: E501
    'google-cloud-storage',
    'matplotlib'
]  # yapf: disable

setup(name='metarl_benchmarks',
      packages=find_packages(where='src'),
      package_dir={'': 'src'},
      install_requires=REQUIRED,
      include_package_data=True,
      entry_points='''
              [console_scripts]
              metarl_benchmark=metarl_benchmarks.run_benchmarks:cli
          ''')
