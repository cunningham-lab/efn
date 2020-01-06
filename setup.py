#!/usr/bin/env python

from setuptools import setup

setup(name='efn',
      version='0.1',
      description='Exponential family networks.',
      author='Sean Bittner',
      author_email='srb2201@columbia.edu',
      packages=['efn', 'efn.util'],
      install_requires=['tensorflow==1.15', 'numpy', 'statsmodels', \
                        'scipy', 'cvxopt', 'matplotlib', 'scikit-learn'],
     )
