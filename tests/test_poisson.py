"""
Poisson distribution testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from pytest import approx, raises

from cprior.models import ExponentialModel
from cprior.models import PoissonModel
from cprior.models import PoissonABTest
from cprior.models import PoissonMVTest


def test_poisson_model_update():
    model = PoissonModel(shape=0.001, rate=0.001)
    data = np.array([5, 4, 4, 4, 2])

    model.update(data)

    assert model.n_samples_ == 5


def test_poisson_model_pppdf_x():
    model = PoissonModel(shape=25, rate=1000)

    assert model.pppdf([-1, 0, 2, 20]) == approx(
        [0, 0.975322095, 0.000316346671, 1.683587039e-48])


def test_poisson_model_stats():
    model = PoissonModel(shape=25, rate=1000)

    assert model.ppmean() == approx(0.025)
    assert model.ppvar() == approx(0.025025)


def test_poisson_ab_check_models():
    modelA = PoissonModel(shape=25, rate=1000)
    modelB = ExponentialModel(shape=25, rate=1000)

    with raises(TypeError):
        PoissonABTest(modelA=modelA, modelB=modelB)


def test_poisson_mv_check_model_input():
    modelA = PoissonModel(shape=25, rate=1000)
    modelB = PoissonModel(shape=25, rate=1000)

    with raises(TypeError):
        PoissonMVTest(models=[modelA, modelB])
