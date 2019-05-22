"""
Beta distribution testing
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from pytest import approx, raises

from cprior._lib.cprior import beta_cprior
from cprior.cdist import BetaModel


def test_beta_small_a0():
    assert beta_cprior(10, 50, 60, 250) == approx(0.7088919200, rel=1e-8)


def test_beta_small_a1():
    assert beta_cprior(600, 550, 52, 60) == approx(0.1223144258, rel=1e-8)


def test_beta_small_b0():
    assert beta_cprior(1000, 900, 1200, 1000) == approx(0.8898254504, rel=1e-8)


def test_beta_small_b1():
    assert beta_cprior(1000, 900, 1200, 800) == approx(0.9999982656, rel=1e-8)


def test_beta_equal_params_integer():
    assert beta_cprior(100, 90, 100, 90) == approx(0.5)


def test_beta_equal_params_float():
    assert beta_cprior(100.1, 90.1, 100.1, 90.1) == approx(0.5)


def test_beta_model_alpha_positive():
    with raises(ValueError):
        model = BetaModel(alpha=-1)


def test_beta_model_beta_positive():
    with raises(ValueError):
        model = BetaModel(beta=0)


def test_beta_model_mean():
    model = BetaModel(alpha=4, beta=6)
    assert model.mean() == approx(0.4)
