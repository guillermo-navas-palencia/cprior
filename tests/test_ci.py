"""
Confidence/credible intervals (CI) methods testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from pytest import approx, raises
from scipy import stats

from cprior.cdist.ci import ci_interval


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
