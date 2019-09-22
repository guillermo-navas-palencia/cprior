"""
Bayesian model with Exponential likelihood.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from ..cdist import GammaABTest
from ..cdist import GammaModel
from ..cdist import GammaMVTest
from ..cdist.utils import check_models
from ..cdist.utils import check_mv_models


class ExponentialModel(GammaModel):
    r"""
    Bayesian model with an exponential likelihood and a gamma prior
    distribution.

    Given data samples :math:`\mathbf{x} = (x_1, \ldots, x_n)`
    from an exponential distribution with parameter :math:`\lambda`, the
    posterior distribution is

    .. math::

        \lambda | \mathbf{x} \sim \mathcal{G}\left(\alpha + n,
        \beta + \sum_{i=1}^n x_i \right).

    with prior parameters :math:`\alpha` (shape) and :math:`\beta` (rate).

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
            Data samples from an exponential distribution.
        """
        x = np.asarray(data)
        n = x.size
        self._shape_posterior += n
        self._rate_posterior += np.sum(x)
        self.n_samples_ += n

    def pppdf(self, x):
        r"""
        Posterior predictive probability density function.

        If :math:`X` follows an exponential distribution with parameter
        :math:`\lambda`, then the posterior predictive probability density
        function is given by

        .. math::

            f(x; \alpha, \beta) = \frac{\alpha \beta^{\alpha}}{
            (\beta + x)^{\alpha + 1}}.

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
        a = self._shape_posterior
        b = self._rate_posterior

        x = np.asarray(x)
        pdf = np.zeros(x.shape)
        idx = (x >= 0)
        x = x[idx]

        logpdf = np.log(a) + a * np.log(b) - (a + 1) * np.log(b + x)

        pdf[idx] = np.exp(logpdf)

        return pdf

    def ppmean(self):
        r"""
        Posterior predictive mean.

        If :math:`X` follows an exponential distribution with parameter
        :math:`\lambda`, then the posterior predictive expected value is given
        by

        .. math::

            \mathrm{E}[X] = \frac{\beta}{\alpha - 1},

        where :math:`\alpha` and :math:`\beta` are the posterior values
        of the parameters.

        Returns
        -------
        mean : float
        """
        a = self._shape_posterior
        b = self._rate_posterior

        if a > 1:
            return b / (a - 1)
        else:
            return np.nan

    def ppvar(self):
        r"""
        Posterior predictive variance.

        If :math:`X` follows a Poisson distribution with parameter
        :math:`\lambda`, then the posterior predictive variance is given by

        .. math::

            \mathrm{Var}[X] = \frac{\alpha \beta^2}{(\alpha - 1)^2
            (\alpha - 2)},

        where :math:`\alpha` and :math:`\beta` are the posterior values
        of the parameters.

        Returns
        -------
        var : float
        """
        a = self._shape_posterior
        b = self._rate_posterior

        if a > 2:
            return a * b ** 2 / ((a - 1) ** 2 * (a - 2))
        else:
            return np.nan


class ExponentialABTest(GammaABTest):
    """
    Exponential A/B test.

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

        check_models(ExponentialModel, modelA, modelB)


class ExponentialMVTest(GammaMVTest):
    """
    Exponential Multivariate test.

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

        check_mv_models(ExponentialModel, models)
