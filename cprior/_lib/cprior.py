"""
Wrapper C/C++ library.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy.ctypeslib as npct
import os.path
import platform

from ctypes import c_double, c_int


# load library
libabspath = os.path.dirname(os.path.abspath(__file__))
system_os = platform.system()
linux_os = (system_os == "Linux" or "CYGWIN" in system_os)

if linux_os:
    cprior = npct.load_library("_cprior.so", libabspath)
else:
    cprior = npct.load_library("cprior.dll", libabspath)


cprior.cpp_beta_cprior.restype = c_double
cprior.cpp_beta_cprior.argtypes = [c_double, c_double, c_double, c_double]


def beta_cprior(a0, b0, a1, b1):
    return cprior.cpp_beta_cprior(a0, b0, a1, b1)


cprior.cpp_beta_binomial_cdf_cprior.restype = c_double
cprior.cpp_beta_binomial_cdf_cprior.argtypes = [c_int, c_int, c_double,
                                                c_double]


def beta_binomial_cdf_cprior(k, n, a, b):
    return cprior.cpp_beta_binomial_cdf_cprior(k, n, a, b)


cprior.cpp_beta_binomial_ppf_cprior.restype = c_int
cprior.cpp_beta_binomial_ppf_cprior.argtypes = [c_double, c_int, c_double,
                                                c_double]


def beta_binomial_ppf_cprior(q, n, a, b):
    return cprior.cpp_beta_binomial_ppf_cprior(q, n, a, b)
