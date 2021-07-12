#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import CUDA_HOME

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"

setup(
    name="tracktron",
    version="0.1.0",
    author="duwt",
    packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.6",
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
