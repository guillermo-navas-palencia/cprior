"""
Gamma distribution testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from pytest import approx, raises

from cprior.cdist import GammaABTest
from cprior.cdist import GammaModel


def test_gamma_model_shape_positive():
    with raises(ValueError):
        model = GammaModel(shape=-0.1)


def test_gamma_model_rate_positive():
    with raises(ValueError):
        model = GammaModel(rate=0.0)


def test_gamma_model_stats():
    model = GammaModel(shape=3, rate=4)
    assert model.mean() == approx(0.75)
    assert model.var() == approx(0.1875)
    assert model.std() == approx(0.43301270189)
    assert model.pdf(0.1) == approx(0.2145024147)
    assert model.pdf(0) == 0
    assert model.pdf(1) == approx(0.5861004444)
    assert model.cdf(0) == 0
    assert model.cdf(1) == approx(0.7618966944)
    assert model.ppf(0.1) == approx(0.2755163)
    assert model.ppf(0.5) == approx(0.6685151)


def test_gamma_model_priors():
    model = GammaModel(shape=0.1, rate=0.2)
    assert model.shape_posterior == model.shape
    assert model.rate_posterior == model.rate


def test_gamma_ab_probability():
    modelA = GammaModel(shape=25, rate=10000)
    modelB = GammaModel(shape=30, rate=10000)
    abtest = GammaABTest(modelA, modelB, 1000000, 42)

    assert abtest.probability(method="exact",
        variant="A") == approx(0.2483087176, rel=1e-8)

    assert abtest.probability(method="MC",
        variant="A") == approx(0.2483087176, rel=1e-2)

    assert abtest.probability(method="exact",
        variant="B") == approx(0.7516912823, rel=1e-8)

    assert abtest.probability(method="MC",
        variant="B") == approx(0.7516912823, rel=1e-2)

    assert abtest.probability(method="exact",
        variant="all") == approx([0.2483087176, 0.7516912823], rel=1e-8)

    assert abtest.probability(method="MC",
        variant="all") == approx([0.2483087176, 0.7516912823], rel=1e-2)


def test_gamma_ab_expected_loss():
    pass