"""
Geometric distribution testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from pytest import approx, raises

from cprior import BernoulliModel
from cprior import GeometricABTest
from cprior import GeometricModel
from cprior import GeometricMVTest


def test_geometric_model_update():
    model = GeometricModel(alpha=1, beta=1)
    data = np.array([1, 2, 4, 2, 3, 3, 5, 4, 1, 4])

    model.update(data)

    assert model.n_samples_ == 10


def test_geometric_pppdf_x():
    model = GeometricModel(alpha=4, beta=6)

    assert model.pppdf([-1, 0, 1, 10]) == approx([0, 0, 0.4, 0.0086687306])


def test_geometric_model_stats():
    model = GeometricModel(alpha=4, beta=6)

    assert model.pppdf(0) == approx(0)
    assert model.pppdf(4) == approx(0.0783216783)
    assert model.ppmean() == approx(3)
    assert model.ppvar() == approx(3)


def test_geometric_model_mean_var():
    model = GeometricModel(alpha=1, beta=6)

    assert np.isnan(model.ppmean())
    assert np.isnan(model.ppvar())


def test_geometric_ab_check_models():
    modelA = BernoulliModel(alpha=1, beta=1)
    modelB = GeometricModel(alpha=1, beta=1)

    with raises(TypeError):
        abtest = GeometricABTest(modelA=modelA, modelB=modelB)


def test_geometric_mv_check_model_input():
    modelA = GeometricModel(alpha=1, beta=1)
    modelB = GeometricModel(alpha=1, beta=1)

    with raises(TypeError):
        mvtest = GeometricMVTest(models=[modelA, modelB])
