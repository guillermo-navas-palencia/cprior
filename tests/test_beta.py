"""
Beta distribution testing
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from pytest import approx, raises

from cprior._lib.cprior import beta_cprior
from cprior.cdist import BetaABTest
from cprior.cdist import BetaModel


def test_beta_small_a0():
    assert beta_cprior(10, 50, 60, 250) == approx(0.7088919200, rel=1e-8)


def test_beta_small_a1():
    assert beta_cprior(600, 550, 52, 60) == approx(0.1223144258, rel=1e-8)


def test_beta_small_b0():
    assert beta_cprior(1000, 900, 1200, 1000) == approx(0.8898254504, rel=1e-8)


def test_beta_small_b1():
    assert beta_cprior(1000, 900, 1200, 800) == approx(0.9999982656, rel=1e-8)


def test_beta_equal_params_integer():
    assert beta_cprior(100, 90, 100, 90) == approx(0.5)


def test_beta_equal_params_float():
    assert beta_cprior(100.1, 90.1, 100.1, 90.1) == approx(0.5)


def test_beta_model_alpha_positive():
    with raises(ValueError):
        model = BetaModel(alpha=-1)


def test_beta_model_beta_positive():
    with raises(ValueError):
        model = BetaModel(beta=0)


def test_beta_model_stats():
    model = BetaModel(alpha=4, beta=6)
    assert model.mean() == approx(0.4)
    assert model.var() == approx(0.02181818181818)
    assert model.std() == approx(0.14770978917519)
    assert model.pdf(0.1) == approx(0.297607)
    assert model.pdf(0) == 0
    assert model.pdf(1) == 0
    assert model.cdf(0.1) == approx(0.00833109)
    assert model.cdf(0) == 0
    assert model.cdf(1) == 1
    assert model.ppf(0.1) == approx(0.2103962)
    assert model.ppf(0.5) == approx(0.3930848)


def test_beta_model_priors():
    model = BetaModel(alpha=4, beta=6)
    assert model.alpha_posterior == model.alpha
    assert model.beta_posterior == model.beta


def test_beta_model_check_method():
    modelA = BetaModel(alpha=40, beta=60)
    modelB = BetaModel(alpha=70, beta=90)
    abtest = BetaABTest(modelA, modelB, 1000000, 42)

    with raises(ValueError):
        abtest.probability(method="new")


def test_beta_model_check_variant():
    modelA = BetaModel(alpha=40, beta=60)
    modelB = BetaModel(alpha=70, beta=90)
    abtest = BetaABTest(modelA, modelB, 1000000, 42)

    with raises(ValueError):
        abtest.probability(variant="C")


def test_beta_model_check_lift():
    modelA = BetaModel(alpha=40, beta=60)
    modelB = BetaModel(alpha=70, beta=90)
    abtest = BetaABTest(modelA, modelB, 1000000, 42)

    with raises(ValueError):
        abtest.probability(method="MC", lift=-0.1)


def test_beta_model_check_lift_no_MC():
    modelA = BetaModel(alpha=40, beta=60)
    modelB = BetaModel(alpha=70, beta=90)
    abtest = BetaABTest(modelA, modelB, 1000000, 42)

    with raises(ValueError):
        abtest.probability(method="exact", lift=0.1)


def test_beta_probability():
    modelA = BetaModel(alpha=40, beta=60)
    modelB = BetaModel(alpha=70, beta=90)
    abtest = BetaABTest(modelA, modelB, 1000000, 42)

    assert abtest.probability(method="exact",
        variant="A") == approx(0.2737282537, rel=1e-8)

    assert abtest.probability(method="MLHS",
        variant="A") == approx(0.2737282537, rel=1e-2)

    assert abtest.probability(method="MC",
        variant="A") == approx(0.2737282537, rel=1e-2)

    assert abtest.probability(method="exact",
        variant="B") == approx(0.7262717462, rel=1e-8)

    assert abtest.probability(method="MLHS",
        variant="B") == approx(0.7262717462, rel=1e-2)

    assert abtest.probability(method="MC",
        variant="B") == approx(0.7262717462, rel=1e-2)

    assert abtest.probability(method="exact",
        variant="all") == approx([0.2737282537, 0.7262717462], rel=1e-8)

    assert abtest.probability(method="MLHS",
        variant="all") == approx([0.2737282537, 0.7262717462], rel=1e-2)

    assert abtest.probability(method="MC",
        variant="all") == approx([0.2737282537, 0.7262717462], rel=1e-2)


def test_beta_expected_loss():
    modelA = BetaModel(alpha=40, beta=60)
    modelB = BetaModel(alpha=70, beta=90)
    abtest = BetaABTest(modelA, modelB, 1000000, 42)

    assert abtest.expected_loss(method="exact",
        variant="A") == approx(0.0481119843, rel=1e-8)

    assert abtest.expected_loss(method="MC",
        variant="A") == approx(0.0481119843, rel=1e-2)

    assert abtest.expected_loss(method="exact",
        variant="B") == approx(0.0106119843, rel=1e-8)

    assert abtest.expected_loss(method="MC",
        variant="B") == approx(0.0106119843, rel=1e-2)

    assert abtest.expected_loss(method="exact",
        variant="all") == approx([0.0481119843, 0.0106119843], rel=1e-8)

    assert abtest.expected_loss(method="MC",
        variant="all") == approx([0.0481119843, 0.0106119843], rel=1e-2)


def test_beta_expected_loss_ci():
    modelA = BetaModel(alpha=4000, beta=6000)
    modelB = BetaModel(alpha=7000, beta=9000)
    abtest = BetaABTest(modelA, modelB, 1000000, 42)

    assert abtest.expected_loss_ci(method="MC",
        variant="A") == approx([0.02644029, 0.04853865], rel=1e-2)

    assert abtest.expected_loss_ci(method="asymptotic",
        variant="A") == approx([0.02644029, 0.04853865], rel=1e-1)

    assert abtest.expected_loss_ci(method="MC",
        variant="B") == approx([-0.04853865, -0.02644029], rel=1e-2)

    assert abtest.expected_loss_ci(method="asymptotic",
        variant="B") == approx([-0.04853865, -0.02644029], rel=1e-1)

    ci = abtest.expected_loss_ci(method="MC", variant="all")
    assert ci[0]  == approx([0.02644029, 0.04853865], rel=1e-2)
    assert ci[1]  == approx([-0.04853865, -0.02644029], rel=1e-2)

    ci = abtest.expected_loss_ci(method="asymptotic", variant="all")
    assert ci[0]  == approx([0.02644029, 0.04853865], rel=1e-1)
    assert ci[1]  == approx([-0.04853865, -0.02644029], rel=1e-1)


def test_beta_expected_loss_relative():
    modelA = BetaModel(alpha=40, beta=60)
    modelB = BetaModel(alpha=70, beta=90)
    abtest = BetaABTest(modelA, modelB, 1000000, 42)

    assert abtest.expected_loss_relative(method="exact",
        variant="A") == approx(0.1105769230, rel=1e-8)

    assert abtest.expected_loss_relative(method="MC",
        variant="A") == approx(0.1105769230, rel=1e-2)

    assert abtest.expected_loss_relative(method="exact",
        variant="B") == approx(-0.0782608695, rel=1e-8)

    assert abtest.expected_loss_relative(method="MC",
        variant="B") == approx(-0.0782608695, rel=1e-2)

    assert abtest.expected_loss_relative(method="exact",
        variant="all") == approx([0.1105769230, -0.0782608695], rel=1e-8)

    assert abtest.expected_loss_relative(method="MC",
        variant="all") == approx([0.1105769230, -0.0782608695], rel=1e-2)


def test_beta_expected_loss_relative_ci():
    modelA = BetaModel(alpha=40, beta=60)
    modelB = BetaModel(alpha=70, beta=90)
    abtest = BetaABTest(modelA, modelB, 1000000, 42)

    assert abtest.expected_loss_relative_ci(method="exact",
        variant="A") == approx((-0.1424559672, 0.4175275191), rel=1e-8)

    assert abtest.expected_loss_relative_ci(method="MC",
        variant="A") == approx((-0.1424559672, 0.4175275191), rel=1e-2)

    assert abtest.expected_loss_relative_ci(method="exact",
        variant="B") == approx((-0.2945463234, 0.1661208775), rel=1e-8)

    assert abtest.expected_loss_relative_ci(method="MC",
        variant="B") == approx((-0.2945463234, 0.1661208775), rel=1e-2)

    ci = abtest.expected_loss_relative_ci(method="exact", variant="all")
    assert ci[0]  == approx([-0.1424559672, 0.4175275191], rel=1e-8)
    assert ci[1]  == approx([-0.2945463234, 0.1661208775], rel=1e-8)

    ci = abtest.expected_loss_relative_ci(method="MC", variant="all")
    assert ci[0]  == approx([-0.1424559672, 0.4175275191], rel=1e-2)
    assert ci[1]  == approx([-0.2945463234, 0.1661208775], rel=1e-2)


def test_beta_expected_loss_relative_ci_large_params():
    modelA = BetaModel(alpha=4000, beta=6000)
    modelB = BetaModel(alpha=7000, beta=9000)
    abtest = BetaABTest(modelA, modelB, 1000000, 42)

    assert abtest.expected_loss_relative_ci(method="MC",
        variant="A") == approx([0.06506319, 0.1233082], rel=1e-2)

    assert abtest.expected_loss_relative_ci(method="asymptotic",
        variant="A") == approx([0.06506319, 0.1233082], rel=1e-1)

    assert abtest.expected_loss_relative_ci(method="MC",
        variant="B") == approx([-0.10977237, -0.06108857], rel=1e-2)

    assert abtest.expected_loss_relative_ci(method="asymptotic",
        variant="B") == approx([-0.10977237, -0.06108857], rel=1e-1)

    ci = abtest.expected_loss_relative_ci(method="asymptotic", variant="all")
    assert ci[0]  == approx([0.06506319, 0.1233082], rel=1e-1)
    assert ci[1]  == approx([-0.10977237, -0.06108857], rel=1e-1)
