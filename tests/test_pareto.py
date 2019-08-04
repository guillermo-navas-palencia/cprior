"""
Pareto distribution testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from pytest import approx, raises

from cprior.cdist import ParetoABTest
from cprior.cdist import ParetoModel
from cprior.cdist import ParetoMVTest


def test_pareto_model_scale_positive():
    with raises(ValueError):
        model = ParetoModel(scale=-0.1)


def test_pareto_model_shape_positive():
    with raises(ValueError):
        model = ParetoModel(shape=0.0)


def test_pareto_model_stats():
    model = ParetoModel(scale=2, shape=3)
    assert model.mean() == approx(3)
    assert model.var() == approx(3)
    assert model.std() == approx(1.73205081)
    assert model.pdf(0.1) == 0
    assert model.pdf(2.0) == approx(1.5)
    assert model.pdf(2.1) == approx(1.2340537)
    assert model.cdf(2.0) == 0
    assert model.ppf(0.1) == approx(2.0714883)
    assert model.ppf(0.5) == approx(2.5198421)


def test_pareto_model_priors():
    model = ParetoModel(scale=2, shape=3)
    assert model.scale_posterior == model.scale
    assert model.shape_posterior == model.shape


def test_pareto_ab_probability():
    modelA = ParetoModel(scale=3, shape=2)
    modelB = ParetoModel(scale=6, shape=4)
    abtest = ParetoABTest(modelA, modelB, 1000000)

    assert abtest.probability(method="exact",
        variant="A") == approx(1/6, rel=1e-8)

    assert abtest.probability(method="MC",
        variant="A") == approx(1/6, rel=1e-1)

    assert abtest.probability(method="exact",
        variant="B") == approx(5/6, rel=1e-8)

    assert abtest.probability(method="MC",
        variant="B") == approx(5/6, rel=1e-1)

    assert abtest.probability(method="exact",
        variant="all") == approx([1/6, 5/6], rel=1e-8)

    assert abtest.probability(method="MC",
        variant="all") == approx([1/6, 5/6], rel=1e-1)


def test_pareto_ab_expected_loss():
    modelA = ParetoModel(scale=3, shape=2)
    modelB = ParetoModel(scale=6, shape=4)
    abtest = ParetoABTest(modelA, modelB, 1000000)

    assert abtest.expected_loss(method="exact",
        variant="A") == approx(3.2, rel=1e-8)

    assert abtest.expected_loss(method="MC",
        variant="A") == approx(3.2, rel=1e-1)

    assert abtest.expected_loss(method="exact",
        variant="B") == approx(1.2, rel=1e-8)

    assert abtest.expected_loss(method="MC",
        variant="B") == approx(1.2, rel=1e-1)

    assert abtest.expected_loss(method="exact",
        variant="all") == approx([3.2, 1.2], rel=1e-8)

    assert abtest.expected_loss(method="MC",
        variant="all") == approx([3.2, 1.2], rel=1e-1)


def test_pareto_ab_expected_loss_ci():
    modelA = ParetoModel(scale=3, shape=2)
    modelB = ParetoModel(scale=6, shape=4)
    abtest = ParetoABTest(modelA, modelB, 1000000)

    assert abtest.expected_loss_ci(method="MC",
        variant="A") == approx([-5.94455017, 8.16547636], rel=1e-1)

    assert abtest.expected_loss_ci(method="MC",
        variant="B") == approx([-8.17746057, 5.89091807], rel=1e-1)

    ci = abtest.expected_loss_ci(method="MC", variant="all")
    assert ci[0] == approx([-5.94455017, 8.16547636], rel=1e-1)
    assert ci[1] == approx([-8.17746057, 5.89091807], rel=1e-1)


def test_pareto_ab_expected_loss_relative():
    modelA = ParetoModel(scale=3, shape=2)
    modelB = ParetoModel(scale=6, shape=4)
    abtest = ParetoABTest(modelA, modelB, 1000000)

    assert abtest.expected_loss_relative(method="exact",
        variant="A") == approx(7/9, rel=1e-8)

    assert abtest.expected_loss_relative(method="MC",
        variant="A") == approx(7/9, rel=1e-1)

    assert abtest.expected_loss_relative(method="exact",
        variant="B") == approx(-0.2, rel=1e-8)

    assert abtest.expected_loss_relative(method="MC",
        variant="B") == approx(-0.2, rel=1e-1)

    assert abtest.expected_loss_relative(method="exact",
        variant="all") == approx([7/9, -0.2], rel=1e-8)

    assert abtest.expected_loss_relative(method="MC",
        variant="all") == approx([7/9, -0.2], rel=1e-1)


def test_pareto_ab_expected_loss_relative_ci():
    modelA = ParetoModel(scale=3, shape=2)
    modelB = ParetoModel(scale=6, shape=4)
    abtest = ParetoABTest(modelA, modelB, 1000000)

    assert abtest.expected_loss_relative_ci(method="MC",
        variant="A") == approx((-0.45309435, 2.21632453), rel=1e-1)

    assert abtest.expected_loss_relative_ci(method="MC",
        variant="B") == approx((-0.68873954, 0.82425856), rel=1e-1)

    ci = abtest.expected_loss_relative_ci(method="MC", variant="all")
    assert ci[0] == approx([-0.45309435, 2.21632453], rel=1e-1)
    assert ci[1] == approx([-0.68873954, 0.82425856], rel=1e-1)


def test_pareto_mv_check_method():
    models = {
        "A": ParetoModel(scale=3, shape=2),
        "B": ParetoModel(scale=6, shape=4)
    }

    mvtest = ParetoMVTest(models)

    with raises(ValueError):
        mvtest.probability(method="new")


def test_pareto_mv_probability():
    modelA = ParetoModel(scale=3, shape=2)
    modelB = ParetoModel(scale=6, shape=4)
    abtest = ParetoABTest(modelA, modelB, 1000000)
    mvtest = ParetoMVTest({"A": modelA, "B": modelB}, 1000000)

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


def test_pareto_probability_vs_all():
    models = {
        "A": ParetoModel(scale=3, shape=2),
        "B": ParetoModel(scale=6, shape=4),
        "C": ParetoModel(scale=7, shape=5)
    }

    mvtest = ParetoMVTest(models, 1000000)

    assert mvtest.probability_vs_all(method="MLHS",
        variant="B") == approx(0.26983118, rel=1e-1)

    assert mvtest.probability_vs_all(method="MC",
        variant="B") == approx(0.26983118, rel=1e-1)


def test_pareto_mv_expected_loss():
    modelA = ParetoModel(scale=3, shape=2)
    modelB = ParetoModel(scale=6, shape=4)
    abtest = ParetoABTest(modelA, modelB, 1000000)
    mvtest = ParetoMVTest({"A": modelA, "B": modelB}, 1000000)

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


def test_pareto_mv_expected_loss_vs_all():
    models = {
        "A": ParetoModel(scale=3, shape=2),
        "B": ParetoModel(scale=6, shape=4),
        "C": ParetoModel(scale=7, shape=5)
    }

    mvtest = ParetoMVTest(models, 1000000)

    assert mvtest.expected_loss_vs_all(method="MC",
        variant="B") == approx(2.53139854, rel=1e-1)


def test_pareto_mv_expected_loss_ci():
    modelA = ParetoModel(scale=3, shape=2)
    modelB = ParetoModel(scale=6, shape=4)
    abtest = ParetoABTest(modelA, modelB, 1000000)
    mvtest = ParetoMVTest({"A": modelA, "B": modelB}, 1000000)

    ab_result = abtest.expected_loss_ci(method="MC", variant="A")
    mv_result = mvtest.expected_loss_ci(method="MC", control="B",
        variant="A")

    assert ab_result == approx(mv_result, rel=1e-1)

    ab_result = abtest.expected_loss_ci(method="MC", variant="B")
    mv_result = mvtest.expected_loss_ci(method="MC", variant="B")

    assert ab_result == approx(mv_result, rel=1e-1)


def test_pareto_mv_expected_loss_relative():
    modelA = ParetoModel(scale=3, shape=2)
    modelB = ParetoModel(scale=6, shape=4)
    abtest = ParetoABTest(modelA, modelB, 1000000)
    mvtest = ParetoMVTest({"A": modelA, "B": modelB}, 1000000)

    ab_result = abtest.expected_loss_relative(method="exact", variant="A")
    mv_result = mvtest.expected_loss_relative(method="exact", control="B",
        variant="A")

    assert ab_result == approx(mv_result, rel=1e-8)

    ab_result = abtest.expected_loss_relative(method="MC", variant="A")
    mv_result = mvtest.expected_loss_relative(method="MC", control="B",
        variant="A")

    assert ab_result == approx(mv_result, rel=1e-1)

    ab_result = abtest.expected_loss_relative(method="exact", variant="B")
    mv_result = mvtest.expected_loss_relative(method="exact", variant="B")

    assert ab_result == approx(mv_result, rel=1e-8)

    ab_result = abtest.expected_loss_relative(method="MC", variant="B")
    mv_result = mvtest.expected_loss_relative(method="MC", variant="B")

    assert ab_result == approx(mv_result, rel=1e-1)


def test_pareto_mv_expected_loss_relative_vs_all():
    models = {
        "A": ParetoModel(scale=3, shape=2),
        "B": ParetoModel(scale=6, shape=4),
        "C": ParetoModel(scale=7, shape=5)
    }

    mvtest = ParetoMVTest(models, 1000000)

    assert mvtest.expected_loss_relative_vs_all(method="MC",
        variant="B") == approx(0.30986700, rel=1e-1)


def test_pareto_mv_expected_loss_ci_relative():
    modelA = ParetoModel(scale=3, shape=2)
    modelB = ParetoModel(scale=6, shape=4)
    abtest = ParetoABTest(modelA, modelB, 1000000)
    mvtest = ParetoMVTest({"A": modelA, "B": modelB}, 1000000)

    ab_result = abtest.expected_loss_relative_ci(method="MC", variant="A")
    mv_result = mvtest.expected_loss_relative_ci(method="MC", control="B",
        variant="A")

    assert ab_result == approx(mv_result, rel=1e-2)

    ab_result = abtest.expected_loss_relative_ci(method="MC", variant="B")
    mv_result = mvtest.expected_loss_relative_ci(method="MC", variant="B")

    assert ab_result == approx(mv_result, rel=1e-2)
