"""
Bayesian model with Poisson likelihood.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from scipy import stats

from .cdist import GammaABTest
from .cdist import GammaModel
from .cdist.utils import check_models


class PoissonModel(GammaModel):
    """
    Bayesian model with a Poisson likelihood and a gamma prior distribution.

    Given data samples :math:`\mathbf{x} = (x_1, \ldots, x_n)`
    from a Poisson distribution with parameter :math:`\lambda`, the posterior
    distribution is

    .. math::

        \\lambda | \\mathbf{x} \\sim \\mathcal{G}\\left(\\alpha + \\sum_{i=1}^n
        x_i, \\beta + n \\right).

    with prior parameters :math:`\\alpha` (shape) and :math:`\\beta` (rate).

    Parameters
    ----------
    name : str (default="")
        Model name.

    shape : float (default=0.001)
        Prior parameter shape.

    rate : float (default=0.001)
        Prior parameter rate.

    Attributes
    ----------
    n_samples_ : int
        Number of samples.
    """
    def __init__(self, name="", shape=0.001, rate=0.001):
        super().__init__(name, shape, rate)

        self.n_samples_ = 0

    def update(self, data):
        """
        Update posterior parameters with new data.
        
        Parameters
        ----------
        data : array-like, shape = (n_samples)
            Data samples from a Poisson distribution.
        """
        x = np.asarray(data)
        n = x.size
        self._shape_posterior += np.sum(x)
        self._rate_posterior += n
        self.n_samples_ += n

    def pppdf(self, x):
        """
        Posterior predictive probability density function.

        If :math:`X` follows a Poisson distribution with parameter
        :math:`\\lambda`, then the posterior predictive probability density
        function is given by

        .. math::

            f(x; \\alpha, \\beta) = \\binom{x + \\alpha - 1}{\\alpha -1}
            \\left(\\frac{\\beta}{\\beta + 1}\\right)^{\\alpha}
            \\left(\\frac{1}{\\beta + 1}\\right)^x,

        where :math:`\\alpha` and :math:`\\beta` are the posterior values
        of the parameters. Note that this is the probability mass function
        of the negative binomial distribution, thus

        .. math::

            X \\sim \\mathcal{NB}\\left(\\alpha,
            \\frac{\\beta}{\\beta + 1}\\right)

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

        p = b / (b + 1)
        return stats.nbinom.pmf(x, a, p)

    def ppmean(self):
        """
        Posterior predictive mean.

        If :math:`X` follows a Poisson distribution with parameter
        :math:`\\lambda`, then the posterior predictive expected value is given
        by

        .. math::

            \\mathrm{E}[X] = \\frac{\\alpha}{\\beta},

        where :math:`\\alpha` and :math:`\\beta` are the posterior values
        of the parameters.

        Returns
        -------
        mean : float
        """
        a = self._shape_posterior
        b = self._rate_posterior

        return a / b

    def ppvar(self):
        """
        Posterior predictive variance.

        If :math:`X` follows a Poisson distribution with parameter
        :math:`\\lambda`, then the posterior predictive variance is given by

        .. math::

            \\mathrm{Var}[X] = \\frac{\\alpha(\\beta + 1)}{\\beta^2},

        where :math:`\\alpha` and :math:`\\beta` are the posterior values
        of the parameters.

        Returns
        -------
        var : float
        """
        a = self._shape_posterior
        b = self._rate_posterior

        return a * (b + 1) / b ** 2


class PoissonABTest(GammaABTest):
    """
    Poisson A/B test.

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

        check_models(PoissonModel, modelA, modelB)
