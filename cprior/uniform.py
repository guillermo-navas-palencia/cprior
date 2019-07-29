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
    """
    Bayesian model with uniform likelihood and a Pareto prior distribution.

    Parameters
    ----------
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
        pass

    def pppdf(self, x):
        pass

    def ppmean(self):
        pass

    def ppvar(self):
        pass


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
