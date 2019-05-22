"""
Beta distribution testing
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from pytest import approx

from cprior._lib.cprior import beta_cprior


def test_beta_small_a0():
    a0 = 10
    b0 = 50
    a1 = 60
    b1 = 250

    beta_cprior(a0, b0, a1, b1) == approx(0.7088919200702957, rel=1e-8)

def test_beta_small_a1():
    a0 = 600
    b0 = 550
    a1 = 52
    b1 = 60

    beta_cprior(a0, b0, a1, b1) == approx(0.12231442584597124, rel=1e-8)
