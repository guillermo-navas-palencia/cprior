"""
Bernoulli distribution testing
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from pytest import approx, raises

from cprior import BernoulliABTest
from cprior import BernoulliModel
from cprior import GeometricModel


def test_bernoulli_model_update():
    model = BernoulliModel(alpha=1, beta=1)
    data = np.array([0, 0, 0, 1, 1])

    model.update(data)

    assert model.n_samples_ == 5
    assert model.n_success_ == 2


def test_bernoulli_model_pppdf_x():
    model = BernoulliModel(alpha=1, beta=1)

    with raises(ValueError):
        model.pppdf(2)


def test_bernoulli_model_stats():
    model = BernoulliModel(alpha=4, beta=6)

    assert model.pppdf(0) == approx(0.6)
    assert model.pppdf(1) == approx(0.4)
    assert model.ppmean() == approx(0.4)
    assert model.ppvar() == approx(0.24)


def test_bernoulli_ab_test_check_models():
    modelA = BernoulliModel(alpha=1, beta=1)
    modelB = GeometricModel(alpha=1, beta=1)

    with raises(TypeError):
        abtest = BernoulliABTest(modelA=modelA, modelB=modelB)
