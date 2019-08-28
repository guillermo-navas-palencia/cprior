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
        """
        Log of the Normal-inverse-gamma probability density function.

        Parameters
        ----------
        x: array-like
            Quantiles.

        sig2 : array-like
            Quantiles.

        Returns
        -------
        logpdf : numpy.ndarray
            Log of the probability density function evaluated at (x, sig2).
        """
        self._check_input(x, sig2)

        logsig2 = np.log(sig2)
        t0 = 0.5 * np.log(self.variance_scale) - 0.9189385332046727
        t1 = self.shape * np.log(self.scale) - special.gammaln(self.shape)
        t2 = -(self.shape + 1.5) * logsig2
        t3 = self.scale + 0.5 * self.variance_scale * (x - self.loc) ** 2

        return t0 + t1 + t2 - t3 / sig2

    def pdf(self, x, sig2):
        """
        Normal-inverse-gamma probability density function.

        Parameters
        ----------
        x: array-like
            Quantiles.

        sig2 : array-like
            Quantiles.

        Returns
        -------
        pdf : numpy.ndarray
            Probability density function evaluated at (x, sig2).
        """
        return np.exp(self.logpdf(x, sig2))

    def logcdf(self, x, sig2):
        """
        Log of the Normal-inverse-gamma cumulative distribution function.

        Parameters
        ----------
        x: array-like
            Quantiles.

        sig2 : array-like
            Quantiles.

        Returns
        -------
        logcdf : numpy.ndarray
            Log of the cumulative distribution function evaluated at (x, sig2).
        """
        self._check_input(x, sig2)

        xu = (self.variance_scale / sig2) ** 0.5 * (x - self.loc)
        t0 = -self.scale / sig2 + self.shape * np.log(self.scale)
        t1 = -(self.shape + 1) * np.log(sig2) - special.gammaln(self.shape)
        t2 = special.log_ndtr(xu)

        return t0 + t1 + t2

    def cdf(self, x, sig2):
        """
        Normal-inverse-gamma cumulative distribution function.

        Parameters
        ----------
        x: array-like
            Quantiles.

        sig2 : array-like
            Quantiles.

        Returns
        -------
        cdf : numpy.ndarray
            Cumulative distribution function evaluated at (x, sig2).
        """
        return np.exp(self.logcdf(x, sig2))

    def rvs(self, size=1, random_state=None):
        """
        Normal-inverse-gamma random variates.

        Parameters
        ----------
        size : int (default=1)
            Number of random variates.

        random_state : int or None (default=None)
            The seed used by the random number generator.

        Returns
        -------
        rvs : numpy.ndarray or scalar
            Random variates of given size (size, 2).
        """
        sig2_rv = stats.invgamma(a=self.shape, scale=self.scale).rvs(size=size,
            random_state=random_state)

        x_rv = stats.norm(loc=self.loc, scale=np.sqrt(sig2_rv /
            self.variance_scale)).rvs(size=size, random_state=random_state)

        return np.c_[x_rv, sig2_rv]

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
