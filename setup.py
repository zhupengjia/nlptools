#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# name:      setup.py
# author:    Pengjia Zhu <zhupengjia@gmail.com>

import os
from setuptools import setup, find_packages
from Cython.Build import cythonize

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name = "ailab",
    version = "0.1.0",
    author = "Pengjia Zhu",
    author_email = "zhupengjia@gmail.com",
    description = ("ailab"),
    license = "Commercial",
    keywords = "ailab",
    url = "",
    packages= find_packages(),
    # entry_points={
        # 'console_scripts': [
            # 'embedding = dtm.embedding:main'
            # ]
        # },
    setup_requires=['pytest-runner' ],
    install_requires=[
        "numpy",
        "Cython",
        "pandas",
        "nameko",
        "requests"
        ],
    tests_require=['pytest'],
    ext_modules = cythonize(["ailab/*/*.pyx", "ailab/*/*/*/*.pyx"]),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
    ],
)
