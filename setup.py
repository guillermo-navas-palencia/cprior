import ast
import io
import os
import platform
import re
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


setup(
    name="cprior",
    version='0.1.0',
    description='Fast conjugate prior for Bayesian statistics',
    author='Guillermo Navas-Palencia',
    author_email='g.navas.palencia@gmail.com',
    packages=find_packages(),
    data_files=data_files,
    platforms='any',
    include_package_data=True,
    license='LGPL'
    )