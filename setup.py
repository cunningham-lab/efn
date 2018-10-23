#!/usr/bin/env python

from setuptools import setup

setup(name='efn',
      version='1.0',
      description='Useful tensorflow libraries.',
      author='Sean Bittner',
      author_email='srb2201@columbia.edu',
      packages=['efn', 'efn.util'],
      install_requires=['tf_util', 'tensorflow', 'numpy', 'statsmodels', \
                        'scipy', 'cvxopt', 'matplotlib', 'scikit-learn'],
      dependency_links=['https://github.com/cunningham-lab/tf_util/tarball/master#egg=tf_util-1.0'],
     )
~                                                                                                                      
~                                                                                                                      
~      
