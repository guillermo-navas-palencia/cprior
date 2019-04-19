"""
Bayesian model with Bernoulli likelihood.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from .cdist import BetaABTest
from .cdist import BetaModel
from .cdist.utils import check_models


class BernoulliModel(BetaModel):
    """
    Bayesian model with a Bernoulli likelihood and a beta prior distribution.

    Given data samples :math:`\mathbf{x} = (x_1, \ldots, x_n)`
    from a Bernoulli distribution with parameter :math:`p`, the posterior
    distribution is

    .. math::

        p | \\mathbf{x} \\sim \\mathcal{B}\\left(\\alpha + \\sum_{i=1}^n x_i,
        \\beta + n - \sum_{i=1}^n x_i\\right).
    
    with prior parameters :math:`\\alpha` and :math:`\\beta`.

    Parameters
    ----------
    alpha : int or float (default=1)
        Prior parameter alpha.

    beta: int or float (default=1)
        Prior parameter beta.

    Attributes
    ----------
    n_success_ : int
        Number of successes.

    n_samples_ : int
        Number of samples.
    """    
    def __init__(self, alpha=1, beta=1):
        super().__init__(alpha, beta)

        self.n_success_ = 0
        self.n_samples_ = 0

    def update(self, data):
        """
        Update posterior parameters with new data samples.
        
        Parameters
        ----------
        data : array-like, shape = (n_samples)
            Data samples from a Bernoulli distribution.
        """
        n = len(data)
        n_success = np.sum(data)
        self._alpha_posterior += n_success
        self._beta_posterior += n - n_success
        self.n_samples_ += n
        self.n_success_ += n_success

    def pppdf(self, x):
        """
        Posterior predictive probability density function.

        If :math:`X` is a Bernoulli trial with parameter
        :math:`p \\sim Beta(\\alpha, \\beta)`, then the posterior predictive
        probability density function is given by

        .. math::

            f(x; \\alpha, \\beta) = \\begin{cases}
                \\frac{\\alpha}{\\alpha+ \\beta} & \\text{if $x = 1$}\\\\
                \\frac{\\beta}{\\alpha+ \\beta} & \\text{if $x = 0$} 
            \\end{cases}

        where :math:`\\alpha` and :math:`\\beta` are the posterior values
        of the parameters.

        Returns
        -------
        pdf : float
            Probability density function evaluated at x.
        """
        if not x in (0, 1):
            raise ValueError("x must follow a Bernoulli distribution.")

        a = self._alpha_posterior
        b = self._beta_posterior

        if x == 0:
            return b / (a + b)
        else:
            return a / (a + b)

    def ppmean(self):
        """
        Posterior predictive mean.

        If :math:`X` is a Bernoulli trial with parameter
        :math:`p \\sim Beta(\\alpha, \\beta)`, then the posterior predictive
        expected value is given by

        .. math::
            
            \\mathrm{E}[X] = \\frac{\\alpha}{\\alpha + \\beta},

        where :math:`\\alpha` and :math:`\\beta` are the posterior values
        of the parameters.

        Returns
        -------
        mean : float
        """
        a = self._alpha_posterior
        b = self._beta_posterior

        return a / (a + b)

    def ppvar(self):
        """
        Posterior predictive variance.

        If :math:`X` is a Bernoulli trial with parameter
        :math:`p \\sim Beta(\\alpha, \\beta)`, then the posterior predictive
        variance is given by

        .. math::
            
            \\mathrm{Var}[X] = \\frac{\\alpha \\beta}{(\\alpha + \\beta)^2},

        where :math:`\\alpha` and :math:`\\beta` are the posterior values
        of the parameters.

        Returns
        -------
        var : float        
        """
        a = self._alpha_posterior
        b = self._beta_posterior

        return a * b / (a + b) ** 2


class BernoulliABTest(BetaABTest):
    """
    Bernoulli A/B test.

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

        check_models(BernoulliModel, modelA, modelB)
