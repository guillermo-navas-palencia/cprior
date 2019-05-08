"""
Bayesian model with Exponential likelihood.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from .cdist import GammaABTest
from .cdist import GammaModel
from .cdist.utils import check_models


class ExponentialModel(GammaModel):
    """
    Bayesian model with an exponential likelihood and a gamma prior
    distribution.

    Given data samples :math:`\mathbf{x} = (x_1, \ldots, x_n)`
    from an exponential distribution with parameter :math:`\lambda`, the
    posterior distribution is

    .. math::

        \\lambda | \\mathbf{x} \\sim \\mathcal{G}\\left(\\alpha + n,
        \\beta + \\sum_{i=1}^n x_i \\right).

    with prior parameters :math:`\\alpha` (shape) and :math:`\\beta` (rate).

    Parameters
    ----------
    shape : float (default=0.001)
        Prior parameter shape.

    rate : float (default=0.001)
        Prior parameter rate.
    """
    def __init__(self, shape=0.001, rate=0.001):
        super().__init__(shape, rate)

    def update(self, data):
        """
        Update posterior parameters with new data.
        
        Parameters
        ----------
        data : array-like, shape = (n_samples)
            Data samples from an exponential distribution.
        """
        self._shape_posterior += len(data)
        self._rate_posterior += np.sum(data)

    def pppdf(self, x):
        """
        """
        pass

    def ppmean(self):
        """
        """
        pass

    def ppvar(self):
        """
        """
        pass


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
