"""
Bayesian model with Poisson likelihood.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

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

        \\lambda | \\mathbf{x} \\sim \\mathcal{G}\\left(\\alpha + \\sum_{i=1}^n x_i,
        \\beta + n \\right).

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
        """
        self._shape_posterior += np.sum(data)
        self._rate_posterior += len(data)


class PoissonABTest(GammaABTest):
    """
    """
    def __init__(self, modelA, modelB, simulations=1000000, random_state=None):
        super().__init__(modelA, modelB, simulations, random_state)

        check_models(PoissonModel, modelA, modelB)
