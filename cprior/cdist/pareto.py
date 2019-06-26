"""
Pareto conjugate prior distribution model.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from scipy import stats

from .base import BayesABTest
from .base import BayesModel
from .utils import check_ab_method


def probability_to_beat(a0, b0, a1, b1):
    """Closed-form probability a Pareto is greater than another."""
    r = a0 / (a0 + a1)

    if b1 > b0:
        return (b0 / b1) ** a0 * (r - 1) + 1
    else:
        return (b1 / b0) ** a1 * r


def expected_loss(a0, b0, a1, b1):
    """Closed-form expectation max(difference Pareto, 0)."""
    if b1 > b0:
        r = (b0 / b1) ** a0
        t = a1 * b1 / (a1 - 1) * (1 - r)
        q = a0 / (a0 - 1) * (b0 - b1 * r)
        s = a0 * b1 / (a0 + a1 - 1) * r / (a1 - 1)
        return t - q + s
    else:
        return a0 * b0 / (a0 + a1 - 1) * (b1 / b0) ** a1 / (a1 - 1)


class ParetoModel(BayesModel):
    """
    Pareto conjugate prior distribution model.

    Parameters
    ----------
    scale : float (default=0.005)
        Prior parameter scale.

    shape : float (default=0.005)
        Prior parameter shape.
    """
    def __init__(self, name="", scale=0.005, shape=0.005):
        super().__init__(name)

        self.scale = scale
        self.shape = shape

        self._scale_posterior = scale
        self._shape_posterior = shape

        if self.scale <= 0:
            raise ValueError("scale must be > 0; got {}.".format(self.scale))

        if self.shape <= 0:
            raise ValueError("shape must be > 0; got {}.".format(self.shape))

    @property
    def scale_posterior(self):
        """
        Posterior parameter scale.

        Returns
        -------
        scale : float
        """
        return self._scale_posterior

    @property
    def shape_posterior(self):
        """
        Posterior parameter shape.

        Returns
        -------
        shape : float
        """
        return self._shape_posterior

    def mean(self):
        """Mean of the posterior distribution."""
        return stats.pareto(b=self._shape_posterior,
            scale=self._scale_posterior).mean()

    def var(self):
        """Variance of the posterior distribution."""
        return stats.pareto(b=self._shape_posterior,
            scale=self._scale_posterior).var()

    def std(self):
        """Standard deviation of the posterior distribution."""
        return stats.pareto(b=self._shape_posterior,
            scale=self._scale_posterior).std()

    def pdf(self, x):
        """
        Probability density function of the posterior distribution.

        Parameters
        ----------
        x : array-like
            Quantiles.

        Returns
        -------
        pdf : numpy.ndarray
           Probability density function evaluated at x.
        """
        return stats.pareto(b=self._shape_posterior,
            scale=self._scale_posterior).pdf(x)

    def cdf(self, x):
        """
        Cumulative distribution function of the posterior distribution.

        Parameters
        ----------
        x : array-like
            Quantiles.

        Returns
        -------
        cdf : numpy.ndarray
            Cumulative distribution function evaluated at x.
        """
        return stats.pareto(b=self._shape_posterior,
            scale=self._scale_posterior).cdf(x)

    def ppf(self, q):
        """
        Percent point function (quantile) of the posterior distribution.

        Parameters
        ----------
        x : array-like
            Lower tail probability.

        Returns
        -------
        ppf : numpy.ndarray
            Quantile corresponding to the lower tail probability q.
        """
        return stats.pareto(b=self._shape_posterior,
            scale=self._scale_posterior).ppf(q)

    def rvs(self, size=1, random_state=None):
        """
        Random variates of the posterior distribution.

        Parameters
        ----------
        size : int (default=1)
            Number of random variates.

        random_state : int or None (default=None)
            The seed used by the random number generator.

        Returns
        -------
        rvs : numpy.ndarray or scalar
            Random variates of given size.
        """
        return stats.pareto(b=self._shape_posterior, scale=self._scale_posterior
            ).rvs(size=size, random_state=random_state)


class ParetoABTest(BayesABTest):
    """
    Bayesian A/B testing with prior pareto distribution.

    Parameters
    ----------
    modelA : object
        The beta model for variant A.

    modelB : object
        The beta model for variant B.

    simulations : int or None (default=1000000)
        Number of Monte Carlo simulations.

    random_state : int or None (default=None)
        The seed used by the random number generator.    
    """
    def __init__(self, modelA, modelB, simulations=None, random_state=None):
        super().__init__(modelA, modelB, simulations, random_state)

    def probability(self, method="exact", variant="A", lift=0):
        """
        Compute the error probability or *chance to beat control*.

        * If ``variant == "A"``, :math:`P[A > B + lift]`
        * If ``variant == "B"``, :math:`P[B > A + lift]`
        * If ``variant == "all"``, both.

        If ``lift`` is positive value, the computation method must be Monte
        Carlo sampling.

        Parameters
        ----------
        method : str (default="exact")
            The method of computation. Options are "exact" and "MC".

        variant : str (default="A")
            The chosen variant. Options are "A", "B", "all".

        lift : float (default=0.0)
           The amount of uplift.

        mlhs_samples : int (default=10000)
            Number of samples for MLHS method.
        """ 
        check_ab_method(method=method, method_options=("exact", "MC"),
            variant=variant, lift=lift)

        if method == "exact":
            bA = self.modelA.scale_posterior
            aA = self.modelA.shape_posterior

            bB = self.modelB.scale_posterior
            aB = self.modelB.shape_posterior            

            if variant == "A":
                return probability_to_beat(aB, bB, aA, bA)
            elif variant == "B":
                return probability_to_beat(aA, bA, aB, bB)
            else:
                return (probability_to_beat(aB, bB, aA, bA),
                    probability_to_beat(aA, bA, aB, bB))
        else:
            xA = self.modelA.rvs(self.simulations, self.random_state)
            xB = self.modelB.rvs(self.simulations, self.random_state)

            if variant == "A":
                return (xA > xB + lift).mean()
            elif variant == "B":
                return (xB > xA + lift).mean()
            else:
                return (xA > xB + lift).mean(), (xB > xA + lift).mean()

    def expected_loss(self, method="exact", variant="A", lift=0):
        r"""
        Compute the expected loss. This is the expected uplift lost by choosing
        a given variant.

        * If ``variant == "A"``, :math:`\mathrm{E}[\max(B - A - lift, 0)]`
        * If ``variant == "B"``, :math:`\mathrm{E}[\max(A - B - lift, 0)]`
        * If ``variant == "all"``, both.

        If ``lift`` is positive value, the computation method must be Monte
        Carlo sampling.

        Parameters
        ----------
        method : str (default="exact")
            The method of computation. Options are "exact" and "MC". 

        variant : str (default="A")
            The chosen variant. Options are "A", "B", "all".

        lift : float (default=0.0)
            The amount of uplift.
        """        
        if method == "exact":
            bA = self.modelA.scale_posterior
            aA = self.modelA.shape_posterior

            bB = self.modelB.scale_posterior
            aB = self.modelB.shape_posterior

            if variant == "A":
                return expected_loss(aA, bA, aB, bB)
            elif variant == "B":
                return expected_loss(aB, bB, aA, bA)
            else:
                return (expected_loss(aA, bA, aB, bB),
                    expected_loss(aB, bB, aA, bA))
        else:
            xA = self.modelA.rvs(self.simulations, self.random_state)
            xB = self.modelB.rvs(self.simulations, self.random_state)

            if variant == "A":
                return np.maximum(xB - xA - lift, 0).mean()
            elif variant == "B":
                return np.maximum(xA - xB - lift, 0).mean()
            else:
                return (np.maximum(xB - xA - lift, 0).mean(),
                    np.maximum(xA - xB - lift, 0).mean())        

    def expected_loss_relative(self, method="exact", variant="A"):
        r"""
        Compute expected relative loss for choosing a variant. This can be seen
        as the negative expected relative improvement or uplift.

        * If ``variant == "A"``, :math:`\mathrm{E}[(B - A) / A]`
        * If ``variant == "B"``, :math:`\mathrm{E}[(A - B) / B]`
        * If ``variant == "all"``, both.

        Parameters
        ----------
        method : str (default="exact")
            The method of computation. Options are "exact" and "MC".

        variant : str (default="A")
            The chosen variant. Options are "A", "B", "all".
        """
        check_ab_method(method=method, method_options=("exact", "MC"),
            variant=variant)

        if method == "exact":
            bA = self.modelA.scale_posterior
            aA = self.modelA.shape_posterior

            bB = self.modelB.scale_posterior
            aB = self.modelB.shape_posterior

            if variant == "A":
                return aB * bB / (aB - 1) * aA / (bA * (aA + 1)) - 1
            elif variant == "B":
                return aA * bA / (aA - 1) * aB / (bB * (aB + 1)) - 1
            else:
                return (aB * bB / (aB - 1) * aA / (bA * (aA + 1)) - 1,
                    aA * bA / (aA - 1) * aB / (bB * (aB + 1)) - 1)
        else:
            xA = self.modelA.rvs(self.simulations, self.random_state)
            xB = self.modelB.rvs(self.simulations, self.random_state)

            if variant == "A":
                return ((xB - xA) / xA).mean()
            elif variant == "B":
                return ((xA - xB) / xB).mean()
            else:
                return (((xB - xA) / xA).mean(), ((xA - xB) / xB).mean())

    def expected_loss_ci(self, method="MC", variant="A", interval_length=0.9):
        pass

    def expected_loss_relative_ci(self, method="MC", variant="A",
        interval_length=0.9):
        pass