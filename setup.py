import os
import platform
import sys

from setuptools import find_packages, setup, Command
from setuptools.command.test import test as TestCommand


class CleanCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


# test suites

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = []

    def run_tests(self):
        # import here, because outside the eggs aren't loaded
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


system_os = platform.system()
linux_os = (system_os == "Linux" or "CYGWIN" in system_os)

if linux_os:
    cprior = "_cprior.so"
elif system_os == "Windows":
    cprior = "cprior.dll"
else:
    raise RuntimeError("Unexpected system {}.".format(system_os))

# copy compiled libraries
if linux_os:
    data_files = [('cprior/_lib/', ['cprior/_lib/'+cprior])]
else:
    data_files = [('cprior\\_lib\\', ['cprior\\_lib\\'+cprior])]


# install requirements
install_requires = [
    'matplotlib>=3.0.3',
    'mpmath>=1.0.0',
    'numpy>=1.15.0',
    'pandas>=0.24.2',
    'scipy>=1.0.0',
    'pytest',
    'coverage'
]


setup(
    name="cprior",
    version="0.3.1",
    description="Fast Bayesian A/B and multivariate testing",
    author="Guillermo Navas-Palencia",
    author_email="g.navas.palencia@gmail.com",
    packages=find_packages(),
    data_files=data_files,
    platforms="any",
    include_package_data=True,
    license="LGPL",
    url="https://github.com/guillermo-navas-palencia/cprior",
    tests_require=['pytest'],
    cmdclass={'clean': CleanCommand, 'test': PyTest},
    python_requires='>=3.5',
    classifiers=[
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7']
    )
