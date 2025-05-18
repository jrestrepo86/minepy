#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="minepy",
    version="0.2",
    description="Mutual information neural estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jrestrepo86/minepy.git",
    author="Juan F. Restrepo",
    author_email="juan.restrepo@uner.edu.ar",
    license="MIT",
    packages=find_packages(exclude=["tests", ".github"]),
    keywords="Mutual-Information Neural-Networks",
    python_requires=">=3.6",
    install_requires=[
        "torch",
        "torch_vision",
        "pytorch_lightning",
        "scikit_learn",
        "seaborn",
        "tqdm",
        "numpy",
        "matplotlib",
        "ray",
        "psutil",
        "pillow",
    ],
    test_suite="nose.collector",
    tests_require=["nose", "nose-cover3"],
    zip_safe=False,
)
