"""
Normal distribution testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from pytest import approx, raises

from cprior.models import LogNormalModel
from cprior.models import NormalABTest
from cprior.models import NormalModel
from cprior.models import NormalMVTest


def test_normal_model_update():
    model = NormalModel()
    data = np.array([0.52582692, -2.27489383, -0.71673081, -1.47078207,
                     0.42341474])
    model.update(data)

    assert model.n_samples_ == 5


def test_normal_model_pppdf_x():
    model = NormalModel(loc=3, variance_scale=4, shape=5, scale=6)

    assert model.pppdf([-1, 0, 2, 10]) == approx(
        [0.013519143151857045, 0.040763844214145466, 0.20422548520374817,
         0.00044905542675510024])


def test_normal_model_stats():
    model = NormalModel(loc=3, variance_scale=4, shape=5, scale=6)

    assert model.ppmean() == 3
    assert model.ppvar() == approx(2.8125)


def test_normal_model_mean_var():
    model = NormalModel()

    assert np.isnan(model.ppmean())
    assert np.isnan(model.ppvar())


def test_normal_ab_check_models():
    modelA = NormalModel()
    modelB = LogNormalModel()

    with raises(TypeError):
        NormalABTest(modelA=modelA, modelB=modelB)


def test_normal_mv_check_model_input():
    modelA = NormalModel()
    modelB = NormalModel()

    with raises(TypeError):
        NormalMVTest(models=[modelA, modelB])
