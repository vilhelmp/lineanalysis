#!/usr/bin/env python
"""Python setup.py file for lineanalysis
"""

from distutils.core import setup

setup(
    name='lineanalysis',
    version='0.1',
    author='Magnus Persson',
    author_email='magnusp@vilhelm.nu',
    packages=['lineanalysis'],
    license='BSD',
    description='Various functions for molecular line analysis.',
#    install_requires=['astropy'],
)
