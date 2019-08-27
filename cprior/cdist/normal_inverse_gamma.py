"""
Normal-inverse-gamma prior distribution model.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from scipy import special
from scipy import stats

from .base import BayesABTest
from .base import BayesModel
from .utils import check_ab_method


class NormalInverseGamma(object):
    """
    https://github.com/scipy/scipy/pull/6739/files/8ba21ec3dae7c05033797a6a730de38fb95ff388#diff-3f67e7fdb1ce6a44c0b49df2da9889c5
    """
    def __init__(self, loc=0, variance_scale=1, shape=1, scale=1):
        self.loc = loc
        self.variance_scale = variance_scale
        self.shape = shape
        self.scale = scale

        self._check_parameters()

    def mean(self):
        pass

    def var(self):
        pass

    def std(self):
        pass

    def logpdf(self, x, sig2):
        pass

    def pdf(self, x, sig2):
        pass

    def logcdf(self, x, sig2):
        pass

    def cdf(self, x, sig2):
        pass

    def ppf(self, q):
        pass

    def rvs(self, size=1, random_state=None):
        pass

    def _check_input(self, x, sig2):
        pass

    def _check_parameters(self):
        pass


class NormalInverseGammaModel(BayesModel):
    """
    Normal-inverse-gamma prior distribution model.

    Parameters
    ----------
    name : str (default="")
        Model name.

    mu : int or float
        Prior parameter location.

    la : int or float
        Prior parameter variance scale.

    shape : int or float
        Prior parameter shape.

    scale : int or float
        Prior parameter scale.
    """
    pass


class NormalInverseGammaABTest(BayesABTest):
    """
    Bayesian A/B testing with prior normal-inverse-gamma distribution.

    Parameters
    ----------
    modelA : object
        The beta model for variant A.

    modelB : object
        The beta model for variant B.

    simulations : int or None (default=1000000)
        Number of Monte Carlo simulations.

    random_state : int or None (default=None)
        The seed used by the random number generator.    
    """
    def __init__(self, modelA, modelB, simulations=None, random_state=None):
        super().__init__(modelA, modelB, simulations, random_state)
