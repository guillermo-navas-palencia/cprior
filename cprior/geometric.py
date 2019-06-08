"""
Bayesian model with geometric likelihood.
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


class GeometricModel(BetaModel):
    r"""
    Bayesian model with geometric likelihood and a beta prior distribution.

    Given data samples :math:`\mathbf{x} = (x_1, \ldots, x_n)`
    from a geometric distribution with parameter :math:`p`, the posterior
    distribution is

    .. math::

        p | \mathbf{x} \sim \mathcal{B}\left(\alpha + n, \beta +
        \sum_{i=1}^n x_i - n \right),

    with prior parameters :math:`\alpha` and :math:`\beta`.

    Parameters
    ----------
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
    def __init__(self, name="", alpha=1, beta=1):
        super().__init__(name, alpha, beta)

        self.n_samples_ = 0

    def update(self, data):
        """
        Update posterior parameters with new data samples.
        
        Parameters
        ----------
        data : array-like, shape = (n_samples)
            Data samples from a geometric distribution.
        """
        x = np.asarray(data)
        n = x.size
        self._alpha_posterior += n
        self._beta_posterior += np.sum(x) - n
        self.n_samples_ += n

    def pppdf(self, x):
        r"""
        Posterior predictive probability density function.

        If :math:`X` follows a geometric distribution with parameter
        :math:`p \sim \mathcal{B}(\alpha, \beta)`, then the posterior
        predictive probability density function is given by

        .. math::

            f(x; \alpha, \beta) = \frac{B(\alpha + 1, \beta + x - 1)}{B(
            \alpha, \beta)},

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
        idx = (k >= 1)
        k = k[idx]

        logpdf = special.betaln(a + 1, b + k - 1) - special.betaln(a, b)

        pdf[idx] = np.exp(logpdf)

        return pdf

    def ppmean(self):
        r"""
        Posterior predictive mean.

        If :math:`X` follows a geometric distribution with parameter
        :math:`\lambda`, then the posterior predictive expected value is given
        by

        .. math::

            \mathrm{E}[X] = \frac{\alpha + \beta - 1}{\alpha - 1},

        where :math:`\alpha` and :math:`\beta` are the posterior values
        of the parameters.

        Returns
        -------
        mean : float
        """
        a = self._alpha_posterior
        b = self._beta_posterior

        if a > 1:
            return (a + b - 1) / (a - 1)
        else:
            return np.nan

    def ppvar(self):
        r"""
        Posterior predictive variance.

        If :math:`X` follows a geometric distribution with parameter
        :math:`p \sim \mathcal{B}(\alpha, \beta)`, then the posterior
        predictive variance is given by

        .. math::

            \mathrm{Var}[X] = \frac{\beta (\alpha + \beta - 1)}{
            (\alpha - 1)^2 (\alpha - 2)},

        where :math:`\alpha` and :math:`\beta` are the posterior values
        of the parameters.

        Returns
        -------
        var : float
        """
        a = self._alpha_posterior
        b = self._beta_posterior

        if a > 2:
            return b * (a + b - 1) / ((a - 1) ** 2 * (a - 2))
        else:
            return np.nan


class GeometricABTest(BetaABTest):
    """
    Geometric A/B test.

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

        check_models(GeometricModel, modelA, modelB)


class GeometricMVTest(BetaMVTest):
    """
    Geometric Multivariate test.

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

        check_mv_models(GeometricModel, models)
