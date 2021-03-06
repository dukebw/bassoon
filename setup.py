"""Installs the Bassoon parameter server.

Note that PyTorch is a dependency, and is not listed as a dependency in the
setup script, in order to prevent automatic installation of PyTorch. Therefore
users have to install PyTorch manually.
"""
from setuptools import setup


setup(
    name='Bassoon',
    version='0.0',
    py_modules=['bassoon.parameter_server',
                'bassoon.test.test_param_server'],
    install_requires=[
        'Click', 'numpy', 'twisted'
    ],
    entry_points='''
        [console_scripts]
        bassoon=bassoon.parameter_server:parameter_server
        test_bassoon=bassoon.test.test_param_server:test_param_server
    ''',
)
