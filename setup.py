#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import plmrf

setup(name='plmrf',
      version='1.0',
      author='Dan Garant',
      url='https://github.com/dgarant/pl-markov-network',
      license='MIT',
      py_modules=['plmrf']
     )

