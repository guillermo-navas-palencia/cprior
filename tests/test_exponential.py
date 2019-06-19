"""
Exponential distribution testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from pytest import approx, raises

from cprior import ExponentialABTest
from cprior import ExponentialModel
from cprior import PoissonModel


def test_exponential_model_update():
    model = ExponentialModel(shape=0.001, rate=0.001)
    data = np.array([0.4692680, 3.0101214, 1.3167456, 0.9129425, 0.1696248])

    model.update(data)

    assert model.n_samples_ == 5


def test_exponential_model_pppdf_x():
    model = ExponentialModel(shape=25, rate=1000)

    assert model.pppdf([-1, 0, 20, 100]) == approx(
        [0, 0.025, 0.0149394821, 0.0020976363])


def test_exponential_model_stats():
    model = ExponentialModel(shape=25, rate=1000)

    assert model.ppmean() == approx(41.66666666)
    assert model.ppvar() == approx(1887.077294)


def test_exponential_model_mean_var():
    model = ExponentialModel(shape=0.001, rate=0.001)

    assert np.isnan(model.ppmean())
    assert np.isnan(model.ppvar())


def test_exponential_ab_check_models():
    modelA = ExponentialModel(shape=25, rate=1000)
    modelB = PoissonModel(shape=25, rate=1000)

    with raises(TypeError):
        abtest = ExponentialABTest(modelA=modelA, modelB=modelB)
