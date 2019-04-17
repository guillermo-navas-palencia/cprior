"""
Bayesian model with Geometric likelihood.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from .cdist import BetaABTest
from .cdist import BetaModel
from .cdist.utils import check_models


class GeometricModel(BetaModel):
    """
    Bayesian model with geometric likelihood and a beta prior distribution.

    Parameters
    ----------
    alpha : int or float (default=1)
        Prior parameter alpha.

    beta: int or float (default=1)
        Prior parameter beta.    
    """
    def __init__(self, alpha=1, beta=1):
        super().__init__(alpha, beta)

    def update(self, data):
        """
        Update posterior parameters with new data samples.
        
        Parameters
        ----------
        data : array-like, shape = (n_samples)
            Data samples from a geometric distribution.
        """
        n = len(data)
        self._alpha_posterior += n
        self._beta_posterior += np.sum(data) - n


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