"""
Pareto distribution testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from pytest import approx, raises

from cprior.cdist import ParetoABTest
from cprior.cdist import ParetoModel
from cprior.cdist import ParetoMVTest


def test_pareto_model_scale_positive():
    with raises(ValueError):
        model = ParetoModel(scale=-0.1)


def test_pareto_model_shape_positive():
    with raises(ValueError):
        model = ParetoModel(shape=0.0)


def test_pareto_model_stats():
    model = ParetoModel(scale=2, shape=3)
    assert model.mean() == approx(3)
    assert model.var() == approx(3)
    assert model.std() == approx(1.73205081)
    assert model.pdf(0.1) == 0
    assert model.pdf(2.0) == approx(1.5)
    assert model.pdf(2.1) == approx(1.2340537)
    assert model.cdf(2.0) == 0
    assert model.ppf(0.1) == approx(2.0714883)
    assert model.ppf(0.5) == approx(2.5198421)

def test_pareto_model_priors():
    model = ParetoModel(scale=2, shape=3)
    assert model.scale_posterior == model.scale
    assert model.shape_posterior == model.shape
