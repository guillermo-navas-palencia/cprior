"""
Negative binomial distribution testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from pytest import approx, raises

from cprior.models import BinomialModel
from cprior.models import NegativeBinomialABTest
from cprior.models import NegativeBinomialModel
from cprior.models import NegativeBinomialMVTest


def test_negative_binomial_r():
    with raises(ValueError):
        NegativeBinomialModel(r=0, alpha=1, beta=1)


def test_negative_binomial_model_update():
    model = NegativeBinomialModel(r=10, alpha=1, beta=1)
    data = np.array([26, 31, 36, 35, 24, 44, 25, 40, 55, 24])

    model.update(data)

    assert model.n_samples_ == 10


def test_negative_binomial_model_pppdf_x():
    model = NegativeBinomialModel(r=10, alpha=4, beta=6)

    assert model.pppdf([-1, 0, 10, 20]) == approx(
        [0, 0.0030959752, 0.0428785607, 0.0239013606])


def test_negative_binomial_model_stats():
    model = NegativeBinomialModel(r=10, alpha=4, beta=6)

    assert model.ppmean() == approx(20)
    assert model.ppvar() == approx(390)


def test_negative_binomial_model_mean_var():
    model = NegativeBinomialModel(r=10, alpha=1, beta=6)

    assert np.isnan(model.ppmean())
    assert np.isnan(model.ppvar())


def test_negative_binomial_ab_check_model():
    modelA = NegativeBinomialModel(r=10, alpha=1, beta=1)
    modelB = BinomialModel(m=10, alpha=1, beta=1)

    with raises(TypeError):
        NegativeBinomialABTest(modelA=modelA, modelB=modelB)


def test_negative_binomial_mv_check_model_input():
    modelA = NegativeBinomialModel(r=10, alpha=1, beta=1)
    modelB = NegativeBinomialModel(r=10, alpha=1, beta=1)

    with raises(TypeError):
        NegativeBinomialMVTest(models=[modelA, modelB])
