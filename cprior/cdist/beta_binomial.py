"""
Beta-binomial conjugate prior distribution model.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from scipy import special
from scipy import stats

from .._lib.cprior import beta_binomial_cdf_cprior
from .._lib.cprior import beta_binomial_ppf_cprior
from .base import BayesABTest
from .base import BayesModel
from .base import BayesMVTest


class BetaBinomial(object):
    """
    Beta-binomial distribution.

    Parameters
    ----------
    n : int
        Number of binomial trials.

    a : float
        Shape parameter.

    b : float
        Shape parameter.
    """
    def __init__(self, n, a, b):
        self.n = n
        self.a = a
        self.b = b

        if self.a <= 0:
            raise ValueError("a must be > 0; got {}.".format(self.a))

        if self.b <= 0:
            raise ValueError("b must be > 0; got {}.".format(self.b))

    def mean(self):
        """
        Mean of the beta-binominal distribution.

        Returns
        -------
        mean : float
        """
        return self.n * self.a / (self.a + self.b)

    def var(self):
        """
        Variance of the beta-binomial distribution.

        Returns
        -------
        var : float
        """
        n = self.n
        a = self.a
        b = self.b

        return n * a * b * (a + b + n) / (a + b) ** 2 / (a + b + 1)

    def std(self):
        """
        Standard deviation of the beta-binomial distribution.

        Returns
        -------
        std : float
        """
        return np.sqrt(self.var())

    def logpmf(self, k):
        """
        Log of the beta-binomial probability mass function.

        Parameters
        ----------
        k: array-like
            Quantiles.

        Returns
        -------
        logpmf : numpy.ndarray
            Log of the probability mass function evaluated at k.
        """
        k = self._check_input(k)

        n = self.n
        a = self.a
        b = self.b

        loggm = special.gammaln(n + 1)
        loggx = special.gammaln(k + 1)
        loggmx = special.gammaln(n - k + 1)

        logcomb = loggm - loggx - loggmx
        logbeta = special.betaln(a + k, b + n - k) - special.betaln(a, b)

        return logcomb + logbeta

    def pmf(self, k):
        """
        Beta-binomial probability mass function.

        Parameters
        ----------
        k: array-like
            Quantiles.

        Returns
        -------
        logpmf : numpy.ndarray
            Log of the probability mass function evaluated at k.
        """
        return np.exp(self.logpmf(k))

    def logcdf(self, k):
        """
        Log of the beta-binomial cumulative distribution function.

        Parameters
        ----------
        k: array-like
            Quantiles.

        Returns
        -------
        logcdf : numpy.ndarray
            Log of the cumulative distribution function evaluated at k.
        """
        return np.log(self.cdf(k))

    def cdf(self, k):
        """
        Beta-binomial cumulative distribution function.

        Parameters
        ----------
        k: array-like
            Quantiles.

        Returns
        -------
        logcdf : numpy.ndarray
            Log of the cumulative distribution function evaluated at k.
        """
        k = self._check_input(k)

        if k.size > 1:
            _cdf = np.vectorize(beta_binomial_cdf_cprior)
            return _cdf(k, self.n, self.a, self.b)
        else:
            return beta_binomial_cdf_cprior(k, self.n, self.a, self.b)

    def ppf(self, q):
        """
        Percent point function (quantile) of the beta-binomial distribution.

        Parameters
        ----------
        q : array-like
            Lower tail probability.

        Returns
        -------
        ppf : numpy.ndarray
            Quantile corresponding to the lower tail probability q.
        """
        q = np.asarray(q)

        if np.any(q < 0) or np.any(q > 1):
            raise ValueError("q must be in [0, 1].")

        if q.size > 1:
            _ppf = np.vectorize(beta_binomial_ppf_cprior)
            return _ppf(q, self.n, self.a, self.b)
        else:
            return beta_binomial_ppf_cprior(q, self.n, self.a, self.b)

    def rvs(self, size=1, random_state=None):
        """
        Beta-binomial random variates.

        Parameters
        ----------
        size : int (default=1)
            Number of random variates.

        random_state : int or None (default=None)
            The seed used by the random number generator.

        Returns
        -------
        rvs : numpy.ndarray
            Random variates of given size.
        """
        p = stats.beta(self.a, self.b).rvs(size=size,
                                           random_state=random_state)

        return stats.binom(self.n, p).rvs(size=size, random_state=random_state)

    def _check_input(self, k):
        k = np.floor(k).astype(np.int)

        if np.any(k < 0):
            raise ValueError("k must be >= 0.")

        return k


class BetaBinomialModel(BayesModel):
    """
    Beta-binomial prior distribution model.

    Parameters
    ----------
    n : int
        Number of binomial trials.

    alpha : int or float (default=1)
        Prior parameter alpha.

    beta: int or float (default=1)
        Prior parameter beta.
    """
    def __init__(self, name="", n=1, alpha=1, beta=1):
        super().__init__(name)

        self.n = n
        self.alpha = alpha
        self.beta = beta

        self._alpha_posterior = alpha
        self._beta_posterior = beta

        if self.n < 0:
            raise ValueError("n must be >= 0; got {}.".format(self.n))

        if self.alpha <= 0:
            raise ValueError("alpha must be > 0; got {}.".format(self.alpha))

        if self.beta <= 0:
            raise ValueError("beta must be > 0; got {}.".format(self.beta))

    @property
    def alpha_posterior(self):
        """
        Posterior parameter alpha.

        Returns
        -------
        alpha : float
        """
        return self._alpha_posterior

    @property
    def beta_posterior(self):
        """
        Posterior parameter beta.

        Returns
        -------
        beta : float
        """
        return self._beta_posterior

    def mean(self):
        """
        Mean of the posterior distribution.

        Returns
        -------
        mean : float
        """
        return BetaBinomial(self.n, self._alpha_posterior,
                            self._beta_posterior).mean()

    def var(self):
        """
        Variance of the posterior distribution.

        Returns
        -------
        var : float
        """
        return BetaBinomial(self.n, self._alpha_posterior,
                            self._beta_posterior).var()

    def std(self):
        """
        Standard deviation of the posterior distribution.

        Returns
        -------
        std : float
        """
        return BetaBinomial(self.n, self._alpha_posterior,
                            self._beta_posterior).std()

    def pmf(self, k):
        """
        Probability mass function of the posterior distribution.

        Parameters
        ----------
        k : array-like
            Quantiles.

        Returns
        -------
        pdf : numpy.ndarray
           Probability mass function evaluated at k.
        """
        return BetaBinomial(self.n, self._alpha_posterior,
                            self._beta_posterior).pmf(k)

    def cdf(self, k):
        """
        Cumulative distribution function of the posterior distribution.

        Parameters
        ----------
        k : array-like
            Quantiles.

        Returns
        -------
        cdf : numpy.ndarray
            Cumulative distribution function evaluated at k.
        """
        return BetaBinomial(self.n, self._alpha_posterior,
                            self._beta_posterior).cdf(k)

    def ppf(self, q):
        """
        Percent point function (quantile) of the posterior distribution.

        Parameters
        ----------
        q : array-like
            Lower tail probability.

        Returns
        -------
        ppf : numpy.ndarray
            Quantile corresponding to the lower tail probability q.
        """
        return BetaBinomial(self.n, self._alpha_posterior,
                            self._beta_posterior).ppf(q)


class BetaBinomialABTest(BayesABTest):
    pass


class BetaBinomialMVTest(BayesMVTest):
    pass
