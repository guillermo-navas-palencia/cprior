"""
Bayesian model with Log-normal likelihood with unknown mean and variance.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from scipy import stats

from ..cdist import NormalInverseGammaABTest
from ..cdist import NormalInverseGammaModel
from ..cdist import NormalInverseGammaMVTest
from ..cdist.utils import check_models
from ..cdist.utils import check_mv_models


class LogNormalModel(NormalInverseGammaModel):
    r"""
    Bayesian model with a log-normal likelihood and a normal-inverse-gamma
    prior distribution. The Bayesian model requires same priors as for the
    normal distribution.

    Given data samples :math:`\mathbf{x} = (x_1, \ldots, x_n)`
    from a log-normal distribution with parameters mean :math:`\mu`, and
    variance :math:`\sigma^2`, the posterior distribution is

    .. math::

        \mu, \sigma^2 | \mathbf{x} \sim \mathcal{N}\Gamma^{-1}\left(\mu_n,
        \lambda_n, \alpha_n, \beta_n\right),

    where,

    .. math::

        \mu_n &= \frac{\lambda \mu_0 + n \bar{x}}{\lambda + n},

        \lambda_n &= \lambda + n,

        \alpha_n &= \alpha + \frac{n}{2},

        \beta_n &= \beta + \frac{1}{2} \left(\sum_{i=1}^n (\log(x_i) -
        \bar{x})^2 + \frac{n \lambda (\bar{x} - \mu_0)^2}{\lambda + n} \right).

    with prior parameters :math:`\mu_0` (loc), :math:`\lambda`
    (variance_scale), :math:`\alpha` (shape) and :math:`\beta` (scale). Note
    that :math:`n \bar{x} = \sum_{i=1}^n \log(x_i)`.

    Parameters
    ----------
    name : str (default="")
        Model name.

    loc : float (default=0.001)
        Prior parameter loc.

    variance_scale : float (default=0.001)
        Prior parameter variance_scale.

    shape : float (default=0.001)
        Prior parameter shape.

    scale : float (default=0.001)
        Prior parameter scale.

    Attributes
    ----------
    n_samples_ : int
        Number of samples.
    """
    def __init__(self, name="", loc=0.001, variance_scale=0.001, shape=0.001,
                 scale=0.001):
        super().__init__(name, loc, variance_scale, shape, scale)

        self.n_samples_ = 0

        # auxiliary variable to compute online variance
        self._x_sum = 0
        self._x2_sum = 0

    def update(self, data):
        """
        Update posterior parameters with new data.

        Parameters
        ----------
        data : array-like, shape = (n_samples)
            Data samples from a normal distribution.
        """
        x = np.log(np.asarray(data))
        n = x.size

        self.n_samples_ += n
        self._x_sum += np.sum(x)
        self._x2_sum += np.sum(x * x)

        # variance scale posterior
        self._variance_scale_posterior += n

        # loc posterior
        t = self.variance_scale * self.loc + self._x_sum
        self._loc_posterior = t / self._variance_scale_posterior

        # shape posterior
        self._shape_posterior += n / 2

        # scale posterior
        t = self.n_samples_ * self.variance_scale
        t /= self._variance_scale_posterior
        m = self._x_sum / self.n_samples_
        q = (m - self.loc) ** 2
        ss = (self._x2_sum / self.n_samples_ - m ** 2) * (self.n_samples_ - 1)

        self._scale_posterior = self.scale + 0.5 * (ss + t * q)

    def pppdf(self, x):
        r"""
        Posterior predictive probability density function.

        If :math:`X` follows a log-normal distribution with parameters
        :math:`\mu` and :math:`\sigma^2`, then the posterior predictive
        probability density function is given by the probability density
        function of the following Student's t-distribution

        .. math::

            t_{2 \alpha}\left(\mu_0,
            \frac{\beta (1 + \lambda^{-1})}{\alpha}\right),

        where :math:`\mu_0`, :math:`\lambda`, :math:`\alpha` and :math:`\beta`
        are the posterior values of the parameters.

        Parameters
        ----------
        x : array-like
            Quantiles.

        Returns
        -------
        pdf : float
            Probability density function evaluated at x.
        """
        df = 2 * self._shape_posterior
        loc = self._loc_posterior
        t = (1 + 1. / self._variance_scale_posterior)
        scale = self._scale_posterior * t / self._shape_posterior

        return stats.t(df=df, loc=loc, scale=scale).pdf(x)

    def ppmean(self):
        r"""
        Posterior predictive mean.

        If :math:`X` follows a log-normal distribution with parameters
        :math:`\mu` and :math:`\sigma^2`, then the posterior predictive
        expected value is given by

        .. math::

            \mathrm{E}[X] = \mu_0,

        where :math:`\mu_0` is the posterior value of the parameter.

        Returns
        -------
        mean : float
        """
        df = 2 * self._shape_posterior
        loc = self._loc_posterior

        if df > 1:
            return loc
        else:
            return np.nan

    def ppvar(self):
        r"""
        Posterior predictive variance.

        If :math:`X` follows a log-normal distribution with parameters
        :math:`\mu` and :math:`\sigma^2`, then the posterior predictive
        variance is given by

        .. math::

            \mathrm{Var}[X] = \frac{\left(\beta(1 + \lambda^{-1})\right)^2}
            {\alpha(\alpha - 1)},

        where :math:`\lambda`, :math:`\alpha` and :math:`\beta` are the
        posterior values of the parameters.

        Returns
        -------
        var : float
        """
        a = self._shape_posterior
        b = self._scale_posterior
        la = self._variance_scale_posterior

        if a > 1:
            return (b * (1 + la ** (-1))) ** 2 / (a * (a - 1))
        else:
            return np.nan


class LogNormalABTest(NormalInverseGammaABTest):
    """
    Log-normal A/B test.

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

        check_models(LogNormalModel, modelA, modelB)


class LogNormalMVTest(NormalInverseGammaMVTest):
    """
    Log-normal Multivariate test.

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

        check_mv_models(LogNormalModel, models)
