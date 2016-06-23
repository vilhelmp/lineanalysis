#!/usr/bin/env python
"""Python setup.py file for lineanalysis
"""

from setuptools import setup, find_packages

setup(
    name='lineanalysis',
    version='0.1',
    author='Magnus Persson',
    author_email='magnusp@vilhelm.nu',
    packages=find_packages(),
    license='BSD',
    description='Various functions for molecular line analysis.',
    #install_requires=[],
)

