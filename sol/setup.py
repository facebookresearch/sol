# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# All rights reserved.

# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from Cython.Build import cythonize
import numpy
setup(
    ext_modules=cythonize("controller_reward_computation.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
