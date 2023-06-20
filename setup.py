#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
from setuptools import setup, find_packages

# Constants
DISTNAME = 'scrapl'
DESCRIPTION = 'Scattering with Random Paths for Machine Learning'
URL = 'https://www.kymat.io'
LICENSE = 'MIT'


# Parse description
with open('README.md', encoding='utf8') as f:
    README = f.read().split('\n')
    LONG_DESCRIPTION = '\n'.join([x for x in README if not x[:3] == '[!['])


# Parse version.py
scrapl_version_spec = importlib.util.spec_from_file_location(
    'scrapl_version', 'scrapl/version.py')
scrapl_version_module = importlib.util.module_from_spec(scrapl_version_spec)
scrapl_version_spec.loader.exec_module(scrapl_version_module)
VERSION = scrapl_version_module.version


# Parse requirements.txt
with open('requirements.txt', 'r') as f:
    REQUIREMENTS = f.read().split('\n')


setup_info = dict(
    # Metadata
    name=DISTNAME,
    version=VERSION,
    author=('Vincent Lostanlen', 'Mathieu Lagrange'),
    url=URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
    license=LICENSE,
    packages=find_packages(exclude=('test',)),
    install_requires=REQUIREMENTS,
    zip_safe=True,
)

setup(**setup_info)
