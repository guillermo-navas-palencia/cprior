"""
Log-normal distribution testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from pytest import approx, raises

from cprior.models import LogNormalABTest
from cprior.models import LogNormalModel
from cprior.models import LogNormalMVTest
from cprior.models import NormalModel


def test_lognormal_model_update():
    model = LogNormalModel()
    data = np.array([1.0194489, 1.01270676, 0.54062099, 0.93877564, 1.4486928])
    model.update(data)

    assert model.n_samples_ == 5


def test_lognormal_model_pppdf_x():
    model = LogNormalModel(loc=3, variance_scale=4, shape=5, scale=6)

    assert model.pppdf([-1, 0, 2, 10]) == approx(
        [0.013519143151857045, 0.040763844214145466, 0.20422548520374817,
         0.00044905542675510024])


def test_lognormal_model_stats():
    model = LogNormalModel(loc=3, variance_scale=4, shape=5, scale=6)

    assert model.ppmean() == 3
    assert model.ppvar() == approx(2.8125)


def test_lognormal_model_mean_var():
    model = LogNormalModel()

    assert np.isnan(model.ppmean())
    assert np.isnan(model.ppvar())


def test_lognormal_ab_check_models():
    modelA = LogNormalModel()
    modelB = NormalModel()

    with raises(TypeError):
        LogNormalABTest(modelA=modelA, modelB=modelB)


def test_lognormal_mv_check_model_input():
    modelA = LogNormalModel()
    modelB = LogNormalModel()

    with raises(TypeError):
        LogNormalMVTest(models=[modelA, modelB])
