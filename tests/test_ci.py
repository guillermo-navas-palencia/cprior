"""
Confidence/credible intervals (CI) methods testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from pytest import approx, raises
from scipy import stats

from cprior.cdist.ci import ci_interval
from cprior.cdist.ci import ci_interval_exact


def test_ci_method():
    x = stats.norm.rvs(size=1000, random_state=42)

    with raises(ValueError):
        ci_interval(x, interval_length=0.9, method="new_method")


def test_ci_interval_length():
    x = stats.norm.rvs(size=1000, random_state=42)

    with raises(ValueError):
        ci_interval(x, interval_length=1.1, method="ETI")

    with raises(ValueError):
        ci_interval(x, interval_length=-0.1, method="ETI")


def test_ci():
    x = stats.norm.rvs(size=int(1e6), random_state=42)

    low, high = ci_interval(x, interval_length=0.95, method="ETI")
    assert [-1.96, 1.96] == approx([low, high], rel=1e-2)

    low, high = ci_interval(x, interval_length=0.95, method="HDI")
    assert [-1.96, 1.96] == approx([low, high], rel=1e-2)


def test_ci_exact_method():
    dist = stats.norm

    with raises(ValueError):
        ci_interval_exact(dist, interval_length=0.9, method="new_method")


def test_ci_exact_interval_length():
    dist = stats.norm

    with raises(ValueError):
        ci_interval_exact(dist, interval_length=1.1, method="ETI")

    with raises(ValueError):
        ci_interval_exact(dist, interval_length=-0.1, method="ETI")


def test_ci_exact_dist():
    dist = None

    with raises(TypeError):
        ci_interval_exact(dist, interval_length=0.9, method="ETI")


def test_ci_exact():
    dist = stats.norm

    low, high = ci_interval_exact(dist, interval_length=0.95, method="ETI")
    assert [-1.959963984540054, 1.959963984540054] == approx(
        [low, high], rel=1e-8)

    low, high = ci_interval_exact(dist, interval_length=0.95, method="HDI")
    assert [-1.959963984540054, 1.959963984540054] == approx(
        [low, high], rel=1e-8)


def test_ci_exact_hdi_bounds():
    dist = stats.beta(4, 10)

    bounds = [(0, 1), (0, 1)]
    low, high = ci_interval_exact(dist, interval_length=0.95, method="HDI",
                                  bounds=bounds)
    assert [0.0744808884427984, 0.5138206804161226] == approx(
        [low, high], rel=1e-8)
