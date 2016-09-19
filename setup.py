#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/19/16 2:11 PM
# @Author  : Jackling 



import sys
__title__ = 'ts_predictor'
__version__ = '0.1.0'
__author__ = 'Jackling Gu'
__author_email__ = 'jackling.gu@gmail.com'
try:
    from setuptools import setup, find_packages
except ImportError:
    print '%s now needs setuptools in order to build.' % __title__
    print 'Install it using your package manager (usually python-setuptools) or via pip \
            (pip install setuptools).'
    sys.exit(1)

setup(
        name=__title__,
        version=__version__,
        author=__author__,
        author_email=__author_email__,
        install_requires=[
            'numpy>=1.11.1',
            'pandas>=0.18.1',
            'scikit-learn>=0.16',
            'keras',
            'theano',
            'statsmodels',
            'patsy'
        ],
        package_dir={__title__: 'lib/%s' % __title__},
        packages=find_packages('lib'),
        test_suite="test",
        zip_safe=False
        )