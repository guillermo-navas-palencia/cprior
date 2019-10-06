"""
Bernoulli distribution testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from pytest import approx, raises

from cprior.models import BernoulliABTest
from cprior.models import BernoulliModel
from cprior.models import BernoulliMVTest
from cprior.models import GeometricModel


def test_bernoulli_model_update():
    model = BernoulliModel(alpha=1, beta=1)
    data = np.array([0, 0, 0, 1, 1])

    model.update(data)

    assert model.n_samples_ == 5
    assert model.n_success_ == 2


def test_bernoulli_model_pppdf_x():
    model = BernoulliModel(alpha=4, beta=6)

    assert model.pppdf([0, 1, 2]) == approx([0.6, 0.4, 0])


def test_bernoulli_model_stats():
    model = BernoulliModel(alpha=4, beta=6)

    assert model.pppdf(0) == approx(0.6)
    assert model.pppdf(1) == approx(0.4)
    assert model.ppmean() == approx(0.4)
    assert model.ppvar() == approx(0.24)


def test_bernoulli_ab_check_models():
    modelA = BernoulliModel(alpha=1, beta=1)
    modelB = GeometricModel(alpha=1, beta=1)

    with raises(TypeError):
        BernoulliABTest(modelA=modelA, modelB=modelB)


def test_bernoulli_mv_check_model_input():
    modelA = BernoulliModel(alpha=1, beta=1)
    modelB = BernoulliModel(alpha=1, beta=1)

    with raises(TypeError):
        BernoulliMVTest(models=[modelA, modelB])


def test_bernoulli_mv_check_control():
    models = {
        "B": BernoulliModel(name="variant 1", alpha=1, beta=1),
        "C": BernoulliModel(name="variant 2", alpha=1, beta=1)
    }

    with raises(ValueError):
        BernoulliMVTest(models=models)


def test_bernoulli_mv_check_update():
    models = {
        "A": BernoulliModel(name="control", alpha=1, beta=1),
        "B": BernoulliModel(name="variant 2", alpha=1, beta=1)
    }

    mvtest = BernoulliMVTest(models=models)

    with raises(ValueError):
        data = np.array([0, 0, 0, 1, 1])
        mvtest.update(data=data, variant="C")
