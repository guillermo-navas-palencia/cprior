"""
Beta distribution testing
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from pytest import approx

from cprior._lib.cprior import beta_cprior


def test_beta_small_a0():
    beta_cprior(10, 50, 60, 350) == approx(0.7088919200702957, rel=1e-8)

def test_beta_small_a1():
    beta_cprior(600, 550, 52, 60) == approx(0.12231442584597124, rel=1e-8)

def test_beta_small_b0():
    beta_cprior(1000, 900, 1200, 1000) == approx(0.8898254504596144, rel=1e-8)

def test_beta_small_b1():
    beta_cprior(1000, 900, 1200, 800) == approx(0.999998265666369, rel=1e-8)

def test_beta_equal_params_integer():
    beta_cprior(100, 90, 100, 90) == approx(0.5, rel=1e-8)

def test_beta_equal_params_float():
    beta_cprior(100.1, 90.1, 100.1, 90.1) == approx(0.5, rel=1e-8)
