"""
Gamma distribution testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from pytest import approx, raises

from cprior.cdist import GammaABTest
from cprior.cdist import GammaModel
from cprior.cdist import GammaMVTest


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
    modelA = GammaModel(shape=25, rate=10000)
    modelB = GammaModel(shape=30, rate=10000)
    abtest = GammaABTest(modelA, modelB, 1000000, 42)

    assert abtest.expected_loss(method="exact",
        variant="A") == approx(0.0006094353823, rel=1e-8)

    assert abtest.expected_loss(method="MC",
        variant="A") == approx(0.0006094353823, rel=1e-2)

    assert abtest.expected_loss(method="exact",
        variant="B") == approx(0.0001094353823, rel=1e-8)

    assert abtest.expected_loss(method="MC",
        variant="B") == approx(0.0001094353823, rel=1e-2)

    assert abtest.expected_loss(method="exact",
        variant="all") == approx([0.0006094353823, 0.0001094353823], rel=1e-8)

    assert abtest.expected_loss(method="MC",
        variant="all") == approx([0.0006094353823, 0.0001094353823], rel=1e-2)


def test_gamma_ab_expected_loss_ci():
    modelA = GammaModel(shape=25, rate=10000)
    modelB = GammaModel(shape=30, rate=10000)
    abtest = GammaABTest(modelA, modelB, 1000000, 42)

    assert abtest.expected_loss_ci(method="MC",
        variant="A") == approx([-0.00071135, 0.00171985], rel=1e-2)

    assert abtest.expected_loss_ci(method="asymptotic",
        variant="A") == approx([-0.00071135, 0.00171985], rel=1e-1)

    assert abtest.expected_loss_ci(method="MC",
        variant="B") == approx([-0.00171985, 0.00071135], rel=1e-2)

    assert abtest.expected_loss_ci(method="asymptotic",
        variant="B") == approx([-0.00171985, 0.00071135], rel=1e-1)

    ci = abtest.expected_loss_ci(method="MC", variant="all")
    assert ci[0] == approx([-0.00071135, 0.00171985], rel=1e-2)
    assert ci[1] == approx([-0.00171985, 0.00071135], rel=1e-2)

    ci = abtest.expected_loss_ci(method="asymptotic", variant="all")
    assert ci[0] == approx([-0.00071135, 0.00171985], rel=1e-1)
    assert ci[1] == approx([-0.00171985, 0.00071135], rel=1e-1)


def test_gamma_ab_expected_loss_relative():
    modelA = GammaModel(shape=25, rate=10000)
    modelB = GammaModel(shape=30, rate=10000)
    abtest = GammaABTest(modelA, modelB, 1000000, 42)

    assert abtest.expected_loss_relative(method="exact",
        variant="A") == approx(0.25, rel=1e-8)

    assert abtest.expected_loss_relative(method="MC",
        variant="A") == approx(0.25, rel=1e-2)

    assert abtest.expected_loss_relative(method="exact",
        variant="B") == approx(-0.1379310344, rel=1e-8)

    assert abtest.expected_loss_relative(method="MC",
        variant="B") == approx(-0.1379310344, rel=1e-2)

    assert abtest.expected_loss_relative(method="exact",
        variant="all") == approx([0.25, -0.1379310344], rel=1e-8)

    assert abtest.expected_loss_relative(method="MC",
        variant="all") == approx([0.25, -0.1379310344], rel=1e-2)


def test_gamma_ab_expected_loss_relative_ci():
    modelA = GammaModel(shape=25, rate=10000)
    modelB = GammaModel(shape=30, rate=10000)
    abtest = GammaABTest(modelA, modelB, 1000000, 42)

    assert abtest.expected_loss_relative_ci(method="exact",
        variant="A") == approx((-0.2302812912, 0.8907846796), rel=1e-8)

    assert abtest.expected_loss_relative_ci(method="MC",
        variant="A") == approx((-0.2302812912, 0.8907846796), rel=1e-2)

    assert abtest.expected_loss_relative_ci(method="exact",
        variant="B") == approx((-0.4711190487, 0.2991759050), rel=1e-8)

    assert abtest.expected_loss_relative_ci(method="MC",
        variant="B") == approx((-0.4711190487, 0.2991759050), rel=1e-2)

    ci = abtest.expected_loss_relative_ci(method="exact", variant="all")
    assert ci[0]  == approx([-0.2302812912, 0.8907846796], rel=1e-8)
    assert ci[1]  == approx([-0.4711190487, 0.2991759050], rel=1e-8)

    ci = abtest.expected_loss_relative_ci(method="MC", variant="all")
    assert ci[0]  == approx([-0.2302812912, 0.8907846796], rel=1e-2)
    assert ci[1]  == approx([-0.4711190487, 0.2991759050], rel=1e-2)


def test_gamma_ab_expected_loss_relative_ci_large_params():
    modelA = GammaModel(shape=2500, rate=1000000)
    modelB = GammaModel(shape=3000, rate=1000000)
    abtest = GammaABTest(modelA, modelB, 1000000, 42)

    assert abtest.expected_loss_relative_ci(method="MC",
        variant="A") == approx((0.1477770747, 0.2547302435), rel=1e-2)

    assert abtest.expected_loss_relative_ci(method="asymptotic",
        variant="A") == approx((0.1477770747, 0.2547302435), rel=1e-1)

    assert abtest.expected_loss_relative_ci(method="MC",
        variant="B") == approx((-0.2030159429, -0.1287506763), rel=1e-2)

    assert abtest.expected_loss_relative_ci(method="asymptotic",
        variant="B") == approx((-0.2030159429, -0.1287506763), rel=1e-1)


def test_gamma_mv_check_method():
    models = {
        "A": GammaModel(shape=25, rate=10000),
        "B": GammaModel(shape=30, rate=10000)
    }

    mvtest = GammaMVTest(models)

    with raises(ValueError):
        mvtest.probability(method="new")


def test_gamma_mv_probability():
    modelA = GammaModel(shape=2500, rate=1000000)
    modelB = GammaModel(shape=3000, rate=1000000)
    abtest = GammaABTest(modelA, modelB, 1000000, 42)
    mvtest = GammaMVTest({"A": modelA, "B": modelB}, 1000000, 42)

    ab_result = abtest.probability(method="exact", variant="A")
    mv_result = mvtest.probability(method="exact", control="B", variant="A")

    assert ab_result == approx(mv_result, rel=1e-8)

    ab_result = abtest.probability(method="MC", variant="A")
    mv_result = mvtest.probability(method="MC", control="B", variant="A")

    assert ab_result == approx(mv_result, rel=1e-2)

    ab_result = abtest.probability(method="exact", variant="B")
    mv_result = mvtest.probability(method="exact", variant="B")

    assert ab_result == approx(mv_result, rel=1e-8)

    ab_result = abtest.probability(method="MC", variant="B")
    mv_result = mvtest.probability(method="MC", variant="B")

    assert ab_result == approx(mv_result, rel=1e-2)


def test_gamma_mv_probability_vs_all():
    models = {
        "A": GammaModel(shape=25, rate=10000),
        "B": GammaModel(shape=30, rate=10000),
        "C": GammaModel(shape=40, rate=11000)
    }

    mvtest = GammaMVTest(models, 1000000, 42)

    assert mvtest.probability_vs_all(method="MLHS",
        variant="B") == approx(0.1968051, rel=1e-2)

    assert mvtest.probability_vs_all(method="MC",
        variant="B") == approx(0.1968051, rel=1e-2)


def test_gamma_mv_expected_loss():
    modelA = GammaModel(shape=2500, rate=1000000)
    modelB = GammaModel(shape=3000, rate=1000000)
    abtest = GammaABTest(modelA, modelB, 1000000, 42)
    mvtest = GammaMVTest({"A": modelA, "B": modelB}, 1000000, 42)

    ab_result = abtest.expected_loss(method="exact", variant="A")
    mv_result = mvtest.expected_loss(method="exact", control="B", variant="A")

    assert ab_result == approx(mv_result, rel=1e-8)

    ab_result = abtest.expected_loss(method="MC", variant="A")
    mv_result = mvtest.expected_loss(method="MC", control="B", variant="A")

    assert ab_result == approx(mv_result, rel=1e-2)

    ab_result = abtest.expected_loss(method="exact", variant="B")
    mv_result = mvtest.expected_loss(method="exact", variant="B")

    assert ab_result == approx(mv_result, rel=1e-8)

    ab_result = abtest.expected_loss(method="MC", variant="B")
    mv_result = mvtest.expected_loss(method="MC", variant="B")

    assert ab_result == approx(mv_result, rel=1e-2)


def test_gamma_mv_expected_loss_vs_all():
    models = {
        "A": GammaModel(shape=25, rate=10000),
        "B": GammaModel(shape=30, rate=10000),
        "C": GammaModel(shape=40, rate=11000)
    }

    mvtest = GammaMVTest(models, 1000000, 42)

    assert mvtest.expected_loss_vs_all(method="MLHS",
        variant="B") == approx(0.0007447680, rel=1e-2)

    assert mvtest.expected_loss_vs_all(method="MC",
        variant="B") == approx(0.0007447680, rel=1e-2)


def test_gamma_mv_expected_loss_ci():
    modelA = GammaModel(shape=2500, rate=1000000)
    modelB = GammaModel(shape=3000, rate=1000000)
    abtest = GammaABTest(modelA, modelB, 1000000, 42)
    mvtest = GammaMVTest({"A": modelA, "B": modelB}, 1000000, 42)

    ab_result = abtest.expected_loss_ci(method="MC", variant="A")
    mv_result = mvtest.expected_loss_ci(method="MC", control="B",
        variant="A")

    assert ab_result == approx(mv_result, rel=1e-2)

    ab_result = abtest.expected_loss_ci(method="asymptotic", variant="A")
    mv_result = mvtest.expected_loss_ci(method="asymptotic", control="B",
        variant="A")

    assert ab_result == approx(mv_result, rel=1e-2)

    ab_result = abtest.expected_loss_ci(method="MC", variant="B")
    mv_result = mvtest.expected_loss_ci(method="MC", variant="B")

    assert ab_result == approx(mv_result, rel=1e-2)

    ab_result = abtest.expected_loss_ci(method="asymptotic", variant="B")
    mv_result = mvtest.expected_loss_ci(method="asymptotic", variant="B")

    assert ab_result == approx(mv_result, rel=1e-2)


def test_gamma_mv_expected_loss_relative():
    modelA = GammaModel(shape=2500, rate=1000000)
    modelB = GammaModel(shape=3000, rate=1000000)
    abtest = GammaABTest(modelA, modelB, 1000000, 42)
    mvtest = GammaMVTest({"A": modelA, "B": modelB}, 1000000, 42)

    ab_result = abtest.expected_loss_relative(method="exact", variant="A")
    mv_result = mvtest.expected_loss_relative(method="exact", control="B",
        variant="A")

    assert ab_result == approx(mv_result, rel=1e-2)

    ab_result = abtest.expected_loss_relative(method="MC", variant="A")
    mv_result = mvtest.expected_loss_relative(method="MC", control="B",
        variant="A")

    assert ab_result == approx(mv_result, rel=1e-2)

    ab_result = abtest.expected_loss_relative(method="exact", variant="B")
    mv_result = mvtest.expected_loss_relative(method="exact", variant="B")

    assert ab_result == approx(mv_result, rel=1e-2)

    ab_result = abtest.expected_loss_relative(method="MC", variant="B")
    mv_result = mvtest.expected_loss_relative(method="MC", variant="B")

    assert ab_result == approx(mv_result, rel=1e-2)


def test_gamma_mv_expected_loss_relative_vs_all():
    models = {
        "A": GammaModel(shape=25, rate=10000),
        "B": GammaModel(shape=30, rate=10000),
        "C": GammaModel(shape=40, rate=11000)
    }

    mvtest = GammaMVTest(models, 1000000, 42)

    assert mvtest.expected_loss_relative_vs_all(method="MLHS",
        variant="B") == approx(0.2616121568, rel=1e-2)

    assert mvtest.expected_loss_relative_vs_all(method="MC",
        variant="B") == approx(0.2616121568, rel=1e-2)


def test_gamma_mv_expected_loss_ci_relative():
    modelA = GammaModel(shape=2500, rate=1000000)
    modelB = GammaModel(shape=3000, rate=1000000)
    abtest = GammaABTest(modelA, modelB, 1000000, 42)
    mvtest = GammaMVTest({"A": modelA, "B": modelB}, 1000000, 42)

    ab_result = abtest.expected_loss_relative_ci(method="exact", variant="A")
    mv_result = mvtest.expected_loss_relative_ci(method="exact", control="B",
        variant="A")

    assert ab_result == approx(mv_result, rel=1e-2)

    ab_result = abtest.expected_loss_relative_ci(method="MC", variant="A")
    mv_result = mvtest.expected_loss_relative_ci(method="MC", control="B",
        variant="A")

    assert ab_result == approx(mv_result, rel=1e-2)

    ab_result = abtest.expected_loss_relative_ci(method="asymptotic",
        variant="A")
    mv_result = mvtest.expected_loss_relative_ci(method="asymptotic",
        control="B", variant="A")

    assert ab_result == approx(mv_result, rel=1e-2)

    ab_result = abtest.expected_loss_relative_ci(method="exact", variant="B")
    mv_result = mvtest.expected_loss_relative_ci(method="exact", variant="B")

    assert ab_result == approx(mv_result, rel=1e-2)

    ab_result = abtest.expected_loss_relative_ci(method="MC", variant="B")
    mv_result = mvtest.expected_loss_relative_ci(method="MC", variant="B")

    assert ab_result == approx(mv_result, rel=1e-2)

    ab_result = abtest.expected_loss_relative_ci(method="asymptotic",
        variant="B")
    mv_result = mvtest.expected_loss_relative_ci(method="asymptotic",
        variant="B")

    assert ab_result == approx(mv_result, rel=1e-2)
