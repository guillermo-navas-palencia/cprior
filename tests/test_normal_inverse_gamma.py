"""
Normal-inverse-gamma distribution testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from pytest import approx, raises

from cprior.cdist import NormalInverseGammaABTest
from cprior.cdist import NormalInverseGammaModel
from cprior.cdist import NormalInverseGammaMVTest


def test_normal_inverse_gamma_model_variance_scale_positive():
    with raises(ValueError):
        model = NormalInverseGammaModel(variance_scale=-0.1)


def test_normal_inverse_gamma_model_shape_positive():
    with raises(ValueError):
        model = NormalInverseGammaModel(shape=-0.1)


def test_normal_inverse_gamma_model_scale_positive():
    with raises(ValueError):
        model = NormalInverseGammaModel(scale=-0.1)


def test_normal_inverse_gamma_model_stats():
    model = NormalInverseGammaModel(loc=2, variance_scale=4, shape=3, scale=5)
    assert model.mean() == approx((2, 2.5))
    assert model.var() == approx((0.625, 6.25))
    assert model.std() == approx((0.7905694150, 2.5))
    assert model.pdf(2, 3) == approx(0.0671352013)
    assert model.pdf(0, 1) == approx(0.0001127176)
    assert model.cdf(2, 3) == approx(0.0728686739)
    assert model.cdf(0, 1) == approx(1.3337446804e-05)

def test_normal_inverse_gamma_model_priors():
    model = NormalInverseGammaModel(loc=2, variance_scale=4, shape=3, scale=5)
    assert model.variance_scale_posterior == model.variance_scale
    assert model.shape_posterior == model.shape
    assert model.scale_posterior == model.scale


def test_normal_inverse_gamma_ab_probability():
    modelA = NormalInverseGammaModel(loc=5.5, variance_scale=3, shape=14,
        scale=5)
    modelB = NormalInverseGammaModel(loc=6.0, variance_scale=4, shape=12,
        scale=9)
    abtest = NormalInverseGammaABTest(modelA, modelB, 1000000)

    assert abtest.probability(method="exact",
        variant="A") == approx((0.1930303459, 0.0304179682), rel=1e-8)

    assert abtest.probability(method="MC",
        variant="A") == approx((0.1930303459, 0.0304179682), rel=1e-1)

    assert abtest.probability(method="exact",
        variant="B") == approx((0.8069696540, 0.9695820317), rel=1e-8)

    assert abtest.probability(method="MC",
        variant="B") == approx((0.8069696540, 0.9695820317), rel=1e-1)

    test = abtest.probability(method="exact", variant="all")
    assert test[0] == approx((0.1930303459, 0.0304179682), rel=1e-8)
    assert test[1] == approx((0.8069696540, 0.9695820317), rel=1e-8)

    test = abtest.probability(method="MC", variant="all")
    assert test[0] == approx((0.1930303459, 0.0304179682), rel=1e-1)
    assert test[1] == approx((0.8069696540, 0.9695820317), rel=1e-1)


def test_normal_inverse_gamma_ab_expected_loss():
    modelA = NormalInverseGammaModel(loc=5.5, variance_scale=3, shape=14,
        scale=5)
    modelB = NormalInverseGammaModel(loc=6.0, variance_scale=4, shape=12,
        scale=9)
    abtest = NormalInverseGammaABTest(modelA, modelB, 1000000)

    assert abtest.expected_loss(method="exact",
        variant="A") == approx((0.5615455003, 0.4364267115), rel=1e-8)

    assert abtest.expected_loss(method="MC",
        variant="A") == approx((0.5615455003, 0.4364267115), rel=1e-1)    

    assert abtest.expected_loss(method="exact",
        variant="B") == approx((0.0615455003, 0.0028602780), rel=1e-8)

    assert abtest.expected_loss(method="MC",
        variant="B") == approx((0.0615455003, 0.0028602780), rel=1e-1)

    test = abtest.expected_loss(method="exact", variant="all")
    assert test[0] == approx((0.5615455003, 0.4364267115), rel=1e-8)
    assert test[1] == approx((0.0615455003, 0.0028602780), rel=1e-8)

    test = abtest.expected_loss(method="MC", variant="all")
    assert test[0] == approx((0.5615455003, 0.4364267115), rel=1e-1)
    assert test[1] == approx((0.0615455003, 0.0028602780), rel=1e-1)


def test_normal_inverse_gamma_ab_expected_loss_ci():
    modelA = NormalInverseGammaModel(loc=5.5, variance_scale=30, shape=140,
        scale=5)
    modelB = NormalInverseGammaModel(loc=6.0, variance_scale=40, shape=120,
        scale=9)
    abtest = NormalInverseGammaABTest(modelA, modelB, 1000000)

    ci = abtest.expected_loss_ci(method="MC", variant="A")
    assert ci[0] == approx([0.40867227, 0.59141024], rel=1e-1)
    assert ci[1] == approx([0.02772771, 0.0526537], rel=1e-1)

    ci = abtest.expected_loss_ci(method="asymptotic", variant="A")
    assert ci[0] == approx([0.40867227, 0.59141024], rel=1e-1)
    assert ci[1] == approx([0.02772771, 0.0526537], rel=1e-1)

    ci = abtest.expected_loss_ci(method="MC", variant="B")
    assert ci[0] == approx([-0.59135194, -0.40868052], rel=1e-1)
    assert ci[1] == approx([-0.05266533, -0.02775073], rel=1e-1)

    ci = abtest.expected_loss_ci(method="asymptotic", variant="B")
    assert ci[0] == approx([-0.59135194, -0.40868052], rel=1e-1)
    assert ci[1] == approx([-0.05266533, -0.02775073], rel=1e-1)

    ci = abtest.expected_loss_ci(method="MC", variant="all")
    assert ci[0][0] == approx([0.40867227, 0.59141024], rel=1e-1)
    assert ci[0][1] == approx([0.02772771, 0.0526537], rel=1e-1)
    assert ci[1][0] == approx([-0.59135194, -0.40868052], rel=1e-1)
    assert ci[1][1] == approx([-0.05266533, -0.02775073], rel=1e-1)

    ci = abtest.expected_loss_ci(method="asymptotic", variant="all")
    assert ci[0][0] == approx([0.40867227, 0.59141024], rel=1e-1)
    assert ci[0][1] == approx([0.02772771, 0.0526537], rel=1e-1)
    assert ci[1][0] == approx([-0.59135194, -0.40868052], rel=1e-1)
    assert ci[1][1] == approx([-0.05266533, -0.02775073], rel=1e-1)


def test_normal_inverse_gamma_ab_expected_loss_relative():
    pass


def test_normal_inverse_gamma_ab_expected_loss_relative_ci():
    pass


def test_normal_inverse_gamma_ab_expected_loss_relative_ci_large_params():
    pass


def test_normal_inverse_gamma_mv_check_method():
    models = {
        "A": NormalInverseGammaModel(loc=5.5, variance_scale=3, shape=14,
            scale=5),
        "B": NormalInverseGammaModel(loc=6.0, variance_scale=4, shape=12,
            scale=9)
    }

    mvtest = NormalInverseGammaMVTest(models)

    with raises(ValueError):
        mvtest.probability(method="new")


def test_normal_inverse_gamma_mv_probability():
    modelA = NormalInverseGammaModel(loc=5.5, variance_scale=3, shape=14,
        scale=5)
    modelB = NormalInverseGammaModel(loc=6.0, variance_scale=4, shape=12,
        scale=9)
    abtest = NormalInverseGammaABTest(modelA, modelB, 1000000)
    mvtest = NormalInverseGammaMVTest({"A": modelA, "B": modelB}, 1000000)

    ab_result = abtest.probability(method="exact", variant="A")
    mv_result = mvtest.probability(method="exact", control="B", variant="A")

    assert ab_result == approx(mv_result, rel=1e-8)

    ab_result = abtest.probability(method="MC", variant="A")
    mv_result = mvtest.probability(method="MC", control="B", variant="A")

    assert ab_result == approx(mv_result, rel=1e-1)

    ab_result = abtest.probability(method="exact", variant="B")
    mv_result = mvtest.probability(method="exact", variant="B")

    assert ab_result == approx(mv_result, rel=1e-8)

    ab_result = abtest.probability(method="MC", variant="B")
    mv_result = mvtest.probability(method="MC", variant="B")

    assert ab_result == approx(mv_result, rel=1e-1)


def test_normal_inverse_gamma_mv_probability_vs_all():
    models = {
        "A": NormalInverseGammaModel(loc=5.5, variance_scale=3, shape=14,
            scale=5),
        "B": NormalInverseGammaModel(loc=6.0, variance_scale=4, shape=12,
            scale=9),
        "C": NormalInverseGammaModel(loc=6.1, variance_scale=5, shape=13,
            scale=6)
    }

    mvtest = NormalInverseGammaMVTest(models)


def test_normal_inverse_gamma_mv_expected_loss():
    modelA = NormalInverseGammaModel(loc=5.5, variance_scale=3, shape=14,
        scale=5)
    modelB = NormalInverseGammaModel(loc=6.0, variance_scale=4, shape=12,
        scale=9)
    abtest = NormalInverseGammaABTest(modelA, modelB, 1000000)
    mvtest = NormalInverseGammaMVTest({"A": modelA, "B": modelB}, 1000000)

    ab_result = abtest.expected_loss(method="exact", variant="A")
    mv_result = mvtest.expected_loss(method="exact", control="B", variant="A")

    assert ab_result == approx(mv_result, rel=1e-8)

    ab_result = abtest.expected_loss(method="MC", variant="A")
    mv_result = mvtest.expected_loss(method="MC", control="B", variant="A")

    assert ab_result == approx(mv_result, rel=1e-1)

    ab_result = abtest.expected_loss(method="exact", variant="B")
    mv_result = mvtest.expected_loss(method="exact", variant="B")

    assert ab_result == approx(mv_result, rel=1e-8)

    ab_result = abtest.expected_loss(method="MC", variant="B")
    mv_result = mvtest.expected_loss(method="MC", variant="B")

    assert ab_result == approx(mv_result, rel=1e-1)


def test_normal_inverse_gamma_mv_expected_loss_vs_all():
    models = {
        "A": NormalInverseGammaModel(loc=5.5, variance_scale=3, shape=14,
            scale=5),
        "B": NormalInverseGammaModel(loc=6.0, variance_scale=4, shape=12,
            scale=9),
        "C": NormalInverseGammaModel(loc=6.1, variance_scale=5, shape=13,
            scale=6)
    }

    mvtest = NormalInverseGammaMVTest(models)


def test_normal_inverse_gamma_mv_expected_loss_ci():
    modelA = NormalInverseGammaModel(loc=5.5, variance_scale=3, shape=14,
        scale=5)
    modelB = NormalInverseGammaModel(loc=6.0, variance_scale=4, shape=12,
        scale=9)
    abtest = NormalInverseGammaABTest(modelA, modelB, 1000000)
    mvtest = NormalInverseGammaMVTest({"A": modelA, "B": modelB}, 1000000)

    ab_result = abtest.expected_loss_ci(method="MC", variant="A")
    mv_result = mvtest.expected_loss_ci(method="MC", control="B",
        variant="A")

    assert ab_result[0] == approx(mv_result[0], rel=1e-1)
    assert ab_result[1] == approx(mv_result[1], rel=1e-1)

    ab_result = abtest.expected_loss_ci(method="asymptotic", variant="A")
    mv_result = mvtest.expected_loss_ci(method="asymptotic", control="B",
        variant="A")

    assert ab_result[0] == approx(mv_result[0], rel=1e-8)
    assert ab_result[1] == approx(mv_result[1], rel=1e-8)

    ab_result = abtest.expected_loss_ci(method="MC", variant="B")
    mv_result = mvtest.expected_loss_ci(method="MC", variant="B")

    assert ab_result[0] == approx(mv_result[0], rel=1e-1)
    assert ab_result[1] == approx(mv_result[1], rel=1e-1)

    ab_result = abtest.expected_loss_ci(method="asymptotic", variant="B")
    mv_result = mvtest.expected_loss_ci(method="asymptotic", variant="B")

    assert ab_result[0] == approx(mv_result[0], rel=1e-8)
    assert ab_result[1] == approx(mv_result[1], rel=1e-8)
