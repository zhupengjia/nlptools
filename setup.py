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
    name = "nlptools",
    version = "0.5.0",
    python_requires=">=3.0",
    author = "Pengjia Zhu",
    author_email = "zhupengjia@gmail.com",
    description = "nlptools",
    license = "MIT License",
    keywords = "nlptools",
    url = "",
    packages= find_packages(),
    # entry_points={
        # 'console_scripts': [
            # 'embedding = dtm.embedding:main'
            # ]
        # },
    install_requires=[
        "numpy",
        #"Cython",
        #"pandas",
        #"nameko",
        'bidict',
        #"requests"
        ],
    long_description=read('README.md'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Utilities",
    ],
)
