"""
Binomial distribution testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from pytest import approx, raises

from cprior.models import BinomialABTest
from cprior.models import BinomialModel
from cprior.models import BinomialMVTest
from cprior.models import GeometricModel


def test_binomial_model_m():
    with raises(ValueError):
        model = BinomialModel(m=-1, alpha=1, beta=1)


def test_binomial_model_update():
    model = BinomialModel(m=10, alpha=1, beta=1)
    data = np.array([7, 4, 2, 5, 5, 4, 6, 7, 2, 4])

    model.update(data)

    assert model.n_samples_ == 10


def test_binomial_model_pppdf_x():
    model = BinomialModel(m=10, alpha=4, beta=6)

    assert model.pppdf([-1, 0, 10, 20]) == approx(
        [0, 0.0325077399, 0.0030959752, 0])


def test_binomial_model_stats():
    model = BinomialModel(m=10, alpha=4, beta=6)

    assert model.pppdf(0) == approx(0.0325077399)
    assert model.pppdf(10) == approx(0.0030959752)
    assert model.pppdf(20) == approx(0)
    assert model.ppmean() == approx(4.0)
    assert model.ppvar() == approx(4.36363636364)


def test_binomial_ab_check_models():
    modelA = BinomialModel(m=10, alpha=1, beta=1)
    modelB = GeometricModel(alpha=1, beta=1)

    with raises(TypeError):
        abtest = BinomialABTest(modelA=modelA, modelB=modelB)


def test_binomial_mv_check_model_input():
    modelA = BinomialModel(m=10, alpha=1, beta=1)
    modelB = BinomialModel(m=10, alpha=1, beta=1)

    with raises(TypeError):
        mvtest = BinomialMVTest(models=[modelA, modelB])
