#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# name:      setup.py
# author:    QIAO Nan <qiaonancn@gmail.com>
# license:   GPL
# created:   2016 Dec 17
# modified:  2017 Aug 03
#

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
    version = "0.0.2",
    author = "Qiao, Nan",
    author_email = "qiaonancn@gmail.com",
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
    ext_modules = cythonize("ailab/*/*.pyx"),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
    ],
)
