#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import mrf

setup(name='pll_markov_network',
      version='1.0',
      author='Dan Garant',
      url='https://github.com/dgarant/pll-markov-network',
      license='MIT',
      py_modules=['mrf']
     )

