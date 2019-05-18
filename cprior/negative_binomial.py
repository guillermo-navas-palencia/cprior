"""
Bayesian model with negative binomial likelihood.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from scipy import special

from .cdist import BetaABTest
from .cdist import BetaModel
from .cdist.utils import check_models


class NegativeBinomialModel(BetaModel):
    """
    Bayesian model with a negative binomial likelihood and a beta prior
    distribution.

    Given data samples :math:`\mathbf{x} = (x_1, \ldots, x_n)`
    from a negative binomial distribution with parameters :math:`r` and
    :math:`p`, the posterior distribution is

    .. math::

        p | \\mathbf{x} \\sim \\mathcal{B}\\left(\\alpha + rn,
        \\beta + \sum_{i=1}^n x_i\\right),
    
    with prior parameters :math:`\\alpha` and :math:`\\beta`.

    Parameters
    ----------
    r : int
        Number of failures.

    alpha : int or float (default=1)
        Prior parameter alpha.

    beta: int or float (default=1)
        Prior parameter beta.

    Attributes
    ----------
    n_samples_ : int
        Number of samples.    
    """
    def __init__(self, r, alpha=1, beta=1):
        super().__init__(alpha, beta)

        self.r = r
        self.n_samples_ = 0

    def update(self, data):
        """
        Update posterior parameters with new data samples.

        Parameters
        ----------
        data : array-like, shape = (n_samples)
            Data samples from a negative binomial distribution.
        """
        n = len(data)
        self._alpha_posterior += self.r * n
        self._beta_posterior += np.sum(data)
        self.n_samples_ += n

    def pppdf(self, x):
        """
        Posterior predictive probability density function.
        
        If :math:`X` follows a negative binomial distribution with parameters
        :math:`r` and :math:`p`, then the posterior predictive probability
        density function is given by

        .. math::

            f(x; r, \\alpha, \\beta) = \\binom{x + r - 1}{r - 1}
            \\frac{B(\\alpha + r, \\beta + x)}{B(\\alpha, \\beta)},


        where :math:`\\alpha` and :math:`\\beta` are the posterior values
        of the parameters.

        Parameters
        ----------
        x : array-like
            Quantiles.

        Returns
        -------
        pdf : float
            Probability density function evaluated at x.
        """
        a = self._shape_posterior
        b = self._rate_posterior
        
        loggxr = special.gammaln(self.r + x)
        loggr = special.gammaln(x + 1)
        loggx = special.gammaln(self.r)

        logcomb = loggxr - loggr - loggx
        logbeta = special.betaln(a + self.r, b + x) - special.betaln(a, b)
        logpdf = logcomb + logbeta
        return np.exp(logpdf)

    def ppmean(self):
        """
        Posterior predictive mean.

        If :math:`X` follows a negative binomial distribution with parameters
        :math:`r` and :math:`p`, then the posterior predictive expected value
        is given by

        .. math::
            
            \\mathrm{E}[X] = r \\frac{\\beta}{\\alpha - 1},

        where :math:`\\alpha` and :math:`\\beta` are the posterior values
        of the parameters.

        Returns
        -------
        mean : float
        """
        a = self._alpha_posterior
        b = self._beta_posterior

        return self.r * b / (a - 1)

    def ppvar(self):
        """
        Posterior predictive variance.

        If :math:`X` follows a negative binomial distribution with parameters
        :math:`r` and :math:`p`, then the posterior predictive variance is
        given by

        .. math::

            \\mathrm{Var}[X] = \\frac{r \\beta (\\alpha + r - 1)(
            \\alpha + \\beta - 1)}{(\\alpha - 1)^2 (\\alpha - 2)},

        where :math:`\\alpha` and :math:`\\beta` are the posterior values
        of the parameters.

        Returns
        -------
        var : float        
        """
        a = self._alpha_posterior
        b = self._beta_posterior

        c = self.r * b

        return c * (self.r + a - 1) * (a + b - 1) / (a - 1) ** 2 / (a - 2)


class NegativeBinomialABTest(BetaABTest):
    """
    Negative binomial A/B test.

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

        check_models(NegativeBinomialModel, modelA, modelB)
