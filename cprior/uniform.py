"""
Bayesian model with uniform likelihood.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from scipy import stats

from .cdist import ParetoABTest
from .cdist import ParetoModel
from .cdist import ParetoMVTest
from .cdist.utils import check_models
from .cdist.utils import check_mv_models


class UniformModel(ParetoModel):
    r"""
    Bayesian model with uniform likelihood and a Pareto prior distribution.

    Given data samples :math:`\mathbf{x} = (x_1, \ldots, x_n)` from a uniform
    distribution with zero lower boundary and upper boundary :math:`\theta`, the
    posterior distribution is

    .. math::

        \theta | \mathbf{x} \sim \mathcal{PA}(\alpha + n, \max(\beta, x_{max}))

    with prior parameters :math:`\alpha` (shape), :math:`\beta` (scale) and
    :math:`x_{max}` is the sample maximum.

    Parameters
    ----------
    name : str (default="")
        Model name.

    scale : float (default=0.005)
        Prior parameter scale.

    shape : float (default=0.005)
        Prior parameter shape.

    Attributes
    ----------
    n_samples_ : int
        Number of samples.
    """
    def __init__(self, name="", scale=0.005, shape=0.005):
        super().__init__(name, scale, shape)

        self.n_samples_ = 0

    def update(self, data):
        """
        Update posterior parameters with new data.

        Parameters
        ----------
        data : array-like, shape = (n_samples)
            Data samples from a uniform distribution.
        """
        x = np.asarray(data)
        n = x.size
        self._shape_posterior += n
        self._scale_posterior = max(self._scale_posterior, x.max())
        self.n_samples_ += n

    def pppdf(self, x):
        r"""
        Posterior predictive probability density function.

        If :math:`X` follows a uniform distribution with zero lower boundary and
        upper boundary :math:`\theta`, the posterior predictive probability
        density function is given by

        .. math::

           f(x; \alpha, \beta) = \begin{cases}
              \frac{\alpha}{(\alpha + 1) \beta}, & 0 < x < \beta,\\
              \frac{\alpha \beta^{\alpha}}{(\alpha + 1)x^{\alpha + 1}}, & x \ge
              \beta
           \end{cases}

        where :math:`\alpha` and :math:`\beta` are the posterior values of the
        parameters.

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
        b = self._scale_posterior

        x = np.asarray(x)
        pdf = np.zeros(x.shape)
        idx = (x >= 0)
        x = x[idx]

        pdf[(x < b)] = a / ((a + 1) * b)
        pdf[(x >= b)] = a * b ** a / ((a + 1) * x ** (a + 1))

        return pdf

    def ppmean(self):
        r"""
        Posterior predictive mean.

        If :math:`X` follows a uniform distribution with zero lower boundary and
        upper boundary :math:`\theta`, the posterior predictive expected value
        is given by

        .. math::

            \mathrm{E}[X] = \frac{\alpha \beta}{2(\alpha - 1)},

        where :math:`\alpha` and :math:`\beta` are the posterior values of the
        parameters.

        Returns
        -------
        mean : float
        """
        a = self._shape_posterior
        b = self._scale_posterior

        if a > 1:
            return a * b / (2 * (a - 1))
        else:
            return np.nan

    def ppvar(self):
        r"""
        Posterior predictive variance.

        If :math:`X` follows a uniform distribution with zero lower boundary and
        upper boundary :math:`\theta`, the posterior predictive variance
        is given by

        .. math::

            \mathrm{Var}[X] = \frac{\alpha (\alpha^2 - 2\alpha + 4)\beta^2}
            {12(\alpha - 1)^2 (\alpha - 2)},

        where :math:`\alpha` and :math:`\beta` are the posterior values of the
        parameters.

        Returns
        -------
        var : float
        """
        a = self._shape_posterior
        b = self._scale_posterior

        if a > 2:
            d = 12 * (a - 1) ** 2 * (a-2)
            return a * b ** 2 * (a ** 2 - 2 * a + 4) / d
        else:
            return np.nan


class UniformABTest(ParetoABTest):
    """
    Uniform A/B test.

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

        check_models(UniformModel, modelA, modelB)


class UniformMVTest(ParetoMVTest):
    """
    Uniform Multivariate test.

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

        check_mv_models(UniformModel, models)
