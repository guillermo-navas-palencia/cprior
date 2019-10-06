"""
Uniform distribution testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from pytest import approx, raises

from cprior.models import UniformABTest
from cprior.models import UniformModel
from cprior.models import UniformMVTest
from cprior.models import PoissonModel


def test_uniform_model_update():
    model = UniformModel(scale=0.005, shape=0.005)
    data = np.array([2.2689884, 2.9851765, 0.2667087, 2.9430999, 3.6529030])

    model.update(data)

    assert model.n_samples_ == 5
    assert model._shape_posterior == approx(5.005)
    assert model._scale_posterior == approx(3.6529030)


def test_uniform_model_pppdf_x():
    model = UniformModel(scale=5, shape=3)

    assert model.pppdf([-1, 0, 3, 10]) == approx([0, 0, 3/20, 3/320])


def test_uniform_model_stats():
    model = UniformModel(scale=5, shape=3)

    assert model.ppmean() == approx(15/4)
    assert model.ppvar() == approx(175/16)


def test_uniform_model_mean_var():
    model = UniformModel(scale=0.005, shape=0.005)

    assert np.isnan(model.ppmean())
    assert np.isnan(model.ppvar())


def test_uniform_ab_check_models():
    modelA = UniformModel()
    modelB = PoissonModel()

    with raises(TypeError):
        UniformABTest(modelA=modelA, modelB=modelB)


def test_uniform_mv_check_model_input():
    modelA = UniformModel()
    modelB = UniformModel()

    with raises(TypeError):
        UniformMVTest(models=[modelA, modelB])
