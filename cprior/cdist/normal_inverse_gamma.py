"""
Normal-inverse-gamma prior distribution model.

References:
    [1] https://github.com/scipy/scipy/pull/6739/files/8ba21ec3dae7c05033797a6a730de38fb95ff388#diff-3f67e7fdb1ce6a44c0b49df2da9889c5
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
    Normal-inverse-gamma distribution.

    Parameters
    ----------
    loc : float, optional (default=0)
        Mean of the distribution.

    variance_scale : float, optional (default=1)
        Scale on the normal distribution prior.

    shape : float, optional (default=1)
        Shape of the distribution.

    scale : float, optional (default=1)
        Scale on the inverse gamma distribution prior.
    """
    def __init__(self, loc=0, variance_scale=1, shape=1, scale=1):
        self.loc = loc
        self.variance_scale = variance_scale
        self.shape = shape
        self.scale = scale

        if self.variance_scale <= 0:
            raise ValueError("variance_scale must be > 0; got {}.".format(
                self.variance_scale))

        if self.shape <= 0:
            raise ValueError("shape must be > 0; got {}.".format(self.shape))

        if self.scale <= 0:
            raise ValueError("scale must be > 0; got {}.".format(self.scale))

    def mean(self):
        """
        Mean of the Normal-inverse-gamma probability

        Returns
        -------
        (x_mean, sig2_mean) : tuple of means.
            Mean of the random variates.
        """
        x_mean = self.loc

        if self.shape > 1:
            sig2_mean = self.scale / (self.shape - 1)
        else:
            sig2_mean = np.nan

        return x_mean, sig2_mean

    def mode(self):
        """
        Mode of the Normal-inverse-gamma probability

        Returns
        -------
        (x_mode, sig2_mode) : tuple of modes.
            Mode of the random variates.
        """
        x_mode = self.loc
        sig2_mode = self.scale / (self.shape + 1.5)

        return x_mode, sig2_mode

    def var(self):
        """
        Variance of the Normal-inverse-gamma probability

        Returns
        -------
        (x_var, sig2_var) : tuple of variances.
            Variance of the random variates.
        """
        if self.shape > 1:
            x_var = self.scale / (self.shape - 1) / self.variance_scale
        else:
            x_var = np.nan

        if self.shape > 2:
            sig2_var = self.scale ** 2 / (self.shape - 1) ** 2
            sig2_var /= (self.shape - 2)
        else:
            sig2_var = np.nan

        return x_var, sig2_var

    def std(self):
        """
        Standard deviation of the Normal-inverse-gamma probability

        Returns
        -------
        (x_std, sig2_std) : tuple of standard deviations.
            Standard deviation of the random variates.
        """
        return tuple(np.sqrt(self.var()))

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
        x, sig2 = self._check_input(x, sig2)

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
        x, sig2 = self._check_input(x, sig2)

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
        x = np.asarray(x)
        sig2 = np.asarray(sig2)

        x_shape, sig2_shape = x.shape, sig2.shape

        if x_shape != sig2_shape:
            raise ValueError("Input variables with inconsistent dimensions. "
                "{} != {}".format(x_shape, sig2_shape))

        if np.any(sig2 <= 0):
            raise ValueError("sig2 must be > 0.")

        return x, sig2


class NormalInverseGammaModel(BayesModel):
    """
    Normal-inverse-gamma prior distribution model.

    Parameters
    ----------
    name : str, optional (default="")
        Model name.

    loc : float, optional (default=0.001)
        Prior parameter location.

    variance_scale : float, optional (default=0.001)
        Prior parameter variance scale.

    shape : float, optional (default=0.001)
        Prior parameter shape.

    scale : float, optional (default=0.001)
        Prior parameter scale.
    """
    def __init__(self, name="", loc=0.001, variance_scale=0.001, shape=0.001,
        scale=0.001):
        super().__init__(name)

        self.loc = loc
        self.variance_scale = variance_scale
        self.shape = shape
        self.scale = scale

        self._loc_posterior = loc
        self._variance_scale_posterior = variance_scale
        self._shape_posterior = shape
        self._scale_posterior = scale

        if self.variance_scale <= 0:
            raise ValueError("variance_scale must be > 0; got {}.".format(
                self.variance_scale))

        if self.shape <= 0:
            raise ValueError("shape must be > 0; got {}.".format(self.shape))

        if self.scale <= 0:
            raise ValueError("scale must be > 0; got {}.".format(self.scale))

    @property
    def loc_posterior(self):
        """
        Posterior parameter mu (location)

        Returns
        -------
        mu : float
        """
        return self._loc_posterior

    @property
    def variance_scale_posterior(self):
        """
        Posterior parameter lambda (variance_scale)

        Returns
        -------
        lambda : float
        """
        return self._variance_scale_posterior

    @property
    def shape_posterior(self):
        """
        Posterior parameter alpha (shape).

        Returns
        -------
        alpha : float
        """
        return self._shape_posterior

    @property
    def scale_posterior(self):
        """
        Posterior parameter beta (scale)

        Returns
        -------
        beta : float
        """
        return self._scale_posterior

    def mean(self):
        """Mean of the posterior distribution."""
        return NormalInverseGamma(loc=self._loc_posterior,
            variance_scale=self._variance_scale_posterior,
            shape=self._shape_posterior, scale=self._scale_posterior).mean()

    def var(self):
        """Variance of the posterior distribution."""
        return NormalInverseGamma(loc=self._loc_posterior,
            variance_scale=self._variance_scale_posterior,
            shape=self._shape_posterior, scale=self._scale_posterior).var()

    def std(self):
        """Standard deviation of the posterior distribution."""
        return NormalInverseGamma(loc=self._loc_posterior,
            variance_scale=self._variance_scale_posterior,
            shape=self._shape_posterior, scale=self._scale_posterior).std()

    def pdf(self, x, sig2):
        """
        Probability density function of the posterior distribution.

        Parameters
        ----------
        x : array-like
            Quantiles.

        sig2 : array-like
            Quantiles.

        Returns
        -------
        pdf : numpy.ndarray
           Probability density function evaluated at (x, sig2).
        """
        return NormalInverseGamma(loc=self._loc_posterior,
            variance_scale=self._variance_scale_posterior,
            shape=self._shape_posterior, scale=self._scale_posterior
            ).pdf(x, sig2)

    def cdf(self, x):
        """
        Cumulative distribution function of the posterior distribution.

        Parameters
        ----------
        x : array-like
            Quantiles.

        sig2 : array-like
            Quantiles.

        Returns
        -------
        cdf : numpy.ndarray
            Cumulative distribution function evaluated at (x, sig2).
        """
        return NormalInverseGamma(loc=self._loc_posterior,
            variance_scale=self._variance_scale_posterior,
            shape=self._shape_posterior, scale=self._scale_posterior
            ).cdf(x, sig2)

    def rvs(self, size=1, random_state=None):
        """
        Random variates of the posterior distribution.

        Parameters
        ----------
        size : int (default=1)
            Number of random variates.

        random_state : int or None (default=None)
            The seed used by the random number generator.

        Returns
        -------
        rvs : numpy.ndarray or scalar
            Random variates of given size.
        """
        return NormalInverseGamma(loc=self._loc_posterior,
            variance_scale=self._variance_scale_posterior,
            shape=self._shape_posterior, scale=self._scale_posterior).rvs(
            size=size, random_state=random_state)


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
