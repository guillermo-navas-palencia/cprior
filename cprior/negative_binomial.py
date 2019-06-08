"""
Bayesian model with negative binomial likelihood.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from scipy import special

from .cdist import BetaABTest
from .cdist import BetaModel
from .cdist import BetaMVTest
from .cdist.utils import check_models
from .cdist.utils import check_mv_models


class NegativeBinomialModel(BetaModel):
    r"""
    Bayesian model with a negative binomial likelihood and a beta prior
    distribution.

    Given data samples :math:`\mathbf{x} = (x_1, \ldots, x_n)`
    from a negative binomial distribution with parameters :math:`r` and
    :math:`p`, the posterior distribution is

    .. math::

        p | \mathbf{x} \sim \mathcal{B}\left(\alpha + rn,
        \beta + \sum_{i=1}^n x_i\right),
    
    with prior parameters :math:`\alpha` and :math:`\beta`.

    Parameters
    ----------
    r : int
        Number of failures.

    name : str (default="")
        Model name.

    alpha : int or float (default=1)
        Prior parameter alpha.

    beta: int or float (default=1)
        Prior parameter beta.

    Attributes
    ----------
    n_samples_ : int
        Number of samples.    
    """
    def __init__(self, r, name="", alpha=1, beta=1):
        super().__init__(name, alpha, beta)

        self.r = r
        self.n_samples_ = 0

        if not isinstance(r, np.int) or r <= 0:
            raise ValueError("r must be a positive integer > 0.")

    def update(self, data):
        """
        Update posterior parameters with new data samples.

        Parameters
        ----------
        data : array-like, shape = (n_samples)
            Data samples from a negative binomial distribution.
        """
        x = np.asarray(data)
        n = x.size
        self._alpha_posterior += self.r * n
        self._beta_posterior += np.sum(x)
        self.n_samples_ += n

    def pppdf(self, x):
        r"""
        Posterior predictive probability density function.
        
        If :math:`X` follows a negative binomial distribution with parameters
        :math:`r` and :math:`p`, then the posterior predictive probability
        density function is given by

        .. math::

            f(x; r, \alpha, \beta) = \binom{x + r - 1}{r - 1}
            \frac{B(\alpha + r, \beta + x)}{B(\alpha, \beta)},


        where :math:`\alpha` and :math:`\beta` are the posterior values
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
        a = self._alpha_posterior
        b = self._beta_posterior

        k = np.floor(x)
        pdf = np.zeros(k.shape)
        idx = (k >= 0)
        k = k[idx]
        
        loggxr = special.gammaln(self.r + k)
        loggr = special.gammaln(k + 1)
        loggx = special.gammaln(self.r)

        logcomb = loggxr - loggr - loggx
        logbeta = special.betaln(a + self.r, b + k) - special.betaln(a, b)
        logpdf = logcomb + logbeta

        pdf[idx] = np.exp(logpdf)

        return pdf

    def ppmean(self):
        r"""
        Posterior predictive mean.

        If :math:`X` follows a negative binomial distribution with parameters
        :math:`r` and :math:`p`, then the posterior predictive expected value
        is given by

        .. math::
            
            \mathrm{E}[X] = r \frac{\beta}{\alpha - 1},

        where :math:`\alpha` and :math:`\beta` are the posterior values
        of the parameters.

        Returns
        -------
        mean : float
        """
        a = self._alpha_posterior
        b = self._beta_posterior

        if a > 1:
            return self.r * b / (a - 1)
        else:
            return np.nan

    def ppvar(self):
        r"""
        Posterior predictive variance.

        If :math:`X` follows a negative binomial distribution with parameters
        :math:`r` and :math:`p`, then the posterior predictive variance is
        given by

        .. math::

            \mathrm{Var}[X] = \frac{r \beta (\alpha + r - 1)(
            \alpha + \beta - 1)}{(\alpha - 1)^2 (\alpha - 2)},

        where :math:`\alpha` and :math:`\beta` are the posterior values
        of the parameters.

        Returns
        -------
        var : float        
        """
        a = self._alpha_posterior
        b = self._beta_posterior

        c = self.r * b

        if a > 2:
            return c * (self.r + a - 1) * (a + b - 1) / (a - 1) ** 2 / (a - 2)
        else:
            return np.nan


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


class NegativeBinomialMVTest(BetaMVTest):
    """
    Negative binomial Multivariate test.

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

        check_mv_models(NegativeBinomialModel, models)
