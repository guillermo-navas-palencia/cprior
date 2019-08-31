"""
Bayesian model with Normal likelihood with unknown mean and variance.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from scipy import stats

from ..cdist import NormalInverseGammaABTest
from ..cdist import NormalInverseGammaModel
from ..cdist import NormalInverseGammaMVTest


class NormalModel(NormalInverseGammaModel):
    def __init__(self, name="", loc=0.001, variance_scale=0.001, shape=0.001,
        scale=0.001):
        super().__init__(name, loc, variance_scale, shape, scale)

        self.n_samples_ = 0

    def update(self, data):
        pass

    def pppdf(self, x, sig2):
        pass

    def ppmean(self):
        pass

    def ppvar(self):
        pass


class NormalABTest(NormalInverseGammaABTest):
    """
    Normal A/B test.

    Parameters
    ----------
    modelA : object
        The control model.

    modelB : object
        The variation model.

    simulations : int or None (default=1000000)
        Number of Monte Carlo simulations.

    random_state : int or None (default=None)
        The seed used by the random number generator.
    """
    def __init__(self, modelA, modelB, simulations=1000000, random_state=None):
        super().__init__(modelA, modelB, simulations, random_state)

        check_models(NormalModel, modelA, modelB)


class NormalMVTest(NormalInverseGammaMVTest):
    """
    Normal Multivariate test.

    Parameters
    ----------
    models: dict
        The control and variations models.

    simulations : int or None (default=1000000)
        Number of Monte Carlo simulations.

    random_state : int or None (default=None)
        The seed used by the random number generator.
    """
    def __init__(self, models, simulations=1000000, random_state=None,
        n_jobs=None):
        super().__init__(models, simulations, random_state, n_jobs)

        check_mv_models(NormalModel, models)
