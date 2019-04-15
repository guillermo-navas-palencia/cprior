"""
Beta conjugate prior distribution model.
"""

# guillermo navas-palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np
import mpmath as mp

from scipy import optimize
from scipy import stats

from .._lib.cprior import beta_cprior
from .base import BayesABTest
from .base import BayesModel
from .utils import check_ab_method


def func_ppf(x, a0, b0, a1, b1, p):
    """Function CDF ratio of beta function for root-finding."""
    mp.mp.dps = 100
    one = mp.mp.one

    c = mp.beta(a0 + a1, b0) / (mp.beta(a0, b0) * mp.beta(a1, b1))
    c *= mp.mpf(x) ** -a1 / a1
    f = mp.hyp3f2(a1, a0 + a1, one - b1, a1 + one, a0 + a1 + b0, one / x)
    return float(one - c * f) - p


class BetaModel(BayesModel):
    """
    Beta conjugate prior distribution model.

    Parameters
    ----------
    alpha : int or float (default=1)
        Prior parameter alpha.

    beta: int or float (default=1)
        Prior parameter beta.
    """
    def __init__(self, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta

        self._alpha_posterior = alpha
        self._beta_posterior = beta

        if self.alpha <= 0:
            raise ValueError("alpha must be > 0; got {}.".format(self.alpha))

        if self.beta <= 0:
            raise ValueError("beta must be > 0; got {}.".format(self.beta))

    @property
    def alpha_posterior(self):
        """
        Posterior parameter alpha.

        Returns
        -------
        alpha : float
        """
        return self._alpha_posterior

    @property
    def beta_posterior(self):
        """
        Posterior parameter beta.

        Returns
        -------
        beta : float
        """        
        return self._beta_posterior

    def mean(self):
        """Mean of the posterior distribution."""
        return stats.beta(self._alpha_posterior, self._beta_posterior).mean()

    def var(self):
        """Variance of the posterior distribution."""
        return stats.beta(self._alpha_posterior, self._beta_posterior).var()

    def std(self):
        """Standard deviation of the posterior distribution."""
        return stats.beta(self._alpha_posterior, self._beta_posterior).std()

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
        return stats.beta(self._alpha_posterior, self._beta_posterior).pdf(x)

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
        return stats.beta(self._alpha_posterior, self._beta_posterior).cdf(x)

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
        return stats.beta(self._alpha_posterior, self._beta_posterior).ppf(x)

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
        return stats.beta(self._alpha_posterior,
            self._beta_posterior).rvs(size=size, random_state=random_state)


class BetaABTest(BayesABTest):
    """
    Bayesian A/B testing with prior beta distribution.

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
        """
        check_ab_method(method=method, method_options=("exact", "MC"),
            variant=variant, lift=lift)

        if method == "exact":
            aA = self.modelA.alpha_posterior
            bA = self.modelA.beta_posterior

            aB = self.modelB.alpha_posterior
            bB = self.modelB.beta_posterior

            if variant == "A":
                return beta_cprior(aB, bB, aA, bA)
            elif variant == "B":
                return beta_cprior(aA, bA, aB, bB)
            else:
                return beta_cprior(aB, bB, aA, bA), beta_cprior(aA, bA, aB, bB)
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
        """
        Compute the expected loss. This is the expected uplift lost by choosing
        a given variant.

        * If ``variant == "A"``, :math:`\\max(B - A - lift, 0)`
        * If ``variant == "B"``, :math:`\\max(A - B - lift, 0)`
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
        check_ab_method(method=method, method_options=("exact", "MC"),
            variant=variant, lift=lift)

        if method == "exact":
            aA = self.modelA.alpha_posterior
            bA = self.modelA.beta_posterior

            aB = self.modelB.alpha_posterior
            bB = self.modelB.beta_posterior        

            if variant == "A":
                ta = aA / (aA + bA) * beta_cprior(aA + 1, bA, aB, bB)
                tb = aB / (aB + bB) * beta_cprior(aA, bA, aB + 1, bB)
                return tb - ta
            elif variant == "B":
                ta = aA / (aA + bA) * beta_cprior(aB, bB, aA + 1, bA)
                tb = aB / (aB + bB) * beta_cprior(aB + 1, bB, aA, bA)
                return ta - tb
            else:
                ta = aA / (aA + bA) * beta_cprior(aA + 1, bA, aB, bB)
                tb = aB / (aB + bB) * beta_cprior(aA, bA, aB + 1, bB)
                maxba = tb - ta

                ta = aA / (aA + bA) * beta_cprior(aB, bB, aA + 1, bA)
                tb = aB / (aB + bB) * beta_cprior(aB + 1, bB, aA, bA)
                maxab = ta - tb

                return maxba, maxab
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
        """
        Compute expected relative loss for choosing a variant. This can be seen
        as the negative expected relative improvement or uplift.

        * If ``variant == "A"``, :math:`\\mathrm{E}[(B - A) / A]`
        * If ``variant == "B"``, :math:`\\mathrm{E}[(A - B) / B]`
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
            aA = self.modelA.alpha_posterior
            bA = self.modelA.beta_posterior

            aB = self.modelB.alpha_posterior
            bB = self.modelB.beta_posterior

            if variant == "A":
                return aB * (aA + bA - 1) / (aB + bB) / (aA - 1) - 1
            elif variant == "B":
                return aA * (aB + bB - 1) / (aA + bA) / (aB - 1) - 1
            else:
                return (aB * (aA + bA - 1) / (aB + bB) / (aA - 1) - 1,
                    aA * (aB + bB - 1) / (aA + bA) / (aB - 1) - 1)
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
        """
        Compute credible intervals on the difference distribution of
        :math:`Z = B-A` and/or :math:`Z = A-B`.

        * If ``variant == "A"``, :math:`Z = B - A`
        * If ``variant == "B"``, :math:`Z = A - B`
        * If ``variant == "all"``, both.

        Parameters
        ----------
        method : str (default="MC")
            The method of computation. Options are "asymptotic" and "MC".  

        variant : str (default="A")
            The chosen variant. Options are "A", "B", "all".

        interval_length : float (default=0.9)
            Compute ``interval_length``\% credible interval. This is a value in
            [0, 1].
        """
        check_ab_method(method=method, method_options=("MC", "asymptotic"),
            variant=variant)

        # check interval length
        lower = (1 - interval_length) / 2
        upper = (1 + interval_length) / 2

        if method == "MC":
            xA = self.modelA.rvs(self.simulations, self.random_state)
            xB = self.modelB.rvs(self.simulations, self.random_state)

            lower *= 100.0
            upper *= 100.0
            
            if variant == "A":
                return np.percentile((xB - xA), [lower, upper])
            elif variant == "B":
                return np.percentile((xA - xB), [lower, upper])
            else:
                return (np.percentile((xB - xA), [lower, upper]),
                    np.percentile((xA - xB), [lower, upper]))
        else:
            aA = self.modelA.alpha_posterior
            bA = self.modelA.beta_posterior

            aB = self.modelB.alpha_posterior
            bB = self.modelB.beta_posterior

            mu = aB / (aB + bB) - aA / (aA + bA)
            varA = aA * bA / (aA + bA) ** 2 / (aA + bA + 1)
            varB = aB * bB / (aB + bB) ** 2 / (aB + bB + 1)
            sigma = np.sqrt(varA + varB)

            if variant == "A":
                return stats.norm(mu, sigma).ppf([lower, upper])
            elif variant == "B":
                return stats.norm(-mu, sigma).ppf([lower, upper])
            else:
                return (stats.norm(mu, sigma).ppf([lower, upper]),
                    stats.norm(-mu, sigma).ppf([lower, upper]))

    def expected_loss_relative_ci(self, method="MC", variant="A",
        interval_length=0.9):
        """
        Compute credible intervals on the relative difference distribution of
        :math:`Z = (B-A)/A` and/or :math:`Z = (A-B)/B`.

        * If ``variant == "A"``, :math:`Z = (B-A)/A`
        * If ``variant == "B"``, :math:`Z = (A-B)/B`
        * If ``variant == "all"``, both.

        Parameters
        ----------
        method : str (default="MC")
            The method of computation. Options are "asymptotic", "exact" and
            "MC".  

        variant : str (default="A")
            The chosen variant. Options are "A", "B", "all".

        interval_length : float (default=0.9)
            Compute ``interval_length``\% credible interval. This is a value in
            [0, 1].        
        """
        check_ab_method(method=method,
            method_options=("asymptotic", "exact", "MC"), variant=variant)

        lower = (1 - interval_length) / 2
        upper = (1 + interval_length) / 2        

        if method == "MC":
            xA = self.modelA.rvs(self.simulations, self.random_state)
            xB = self.modelB.rvs(self.simulations, self.random_state)

            lower *= 100.0
            upper *= 100.0

            if variant == "A":
                return np.percentile((xB - xA)/xA, [lower, upper])
            elif variant == "B":
                return np.percentile((xA - xB)/xB, [lower, upper])
            else:
                return (np.percentile((xB - xA)/xA, [lower, upper]),
                    np.percentile((xA - xB)/xB, [lower, upper]))
        else:
            # compute asymptotic
            aA = self.modelA.alpha_posterior
            bA = self.modelA.beta_posterior

            aB = self.modelB.alpha_posterior
            bB = self.modelB.beta_posterior

            if variant == "A":
                mu = aB * (aA + bA - 1) / (aB + bB) / (aA - 1)
                mup = (aB + 1) * (aA + bA - 2) / (aB + bB + 1) / (aA - 2)
                var = mu * (mup - mu)
                sigma = np.sqrt(var)

                dist = stats.norm(mu, sigma)
                ppfl, ppfu = dist.ppf([lower, upper])

                if method == "asymptotic":
                    return ppfl - 1, ppfu - 1
                else:
                    ppfl = optimize.newton(func=func_ppf, x0=ppfl,
                        args=(aB, bB, aA, bA, lower), maxiter=100)

                    ppfu = optimize.newton(func=func_ppf, x0=ppfu,
                        args=(aB, bB, aA, bA, upper), maxiter=100)

                    return ppfl - 1, ppfu - 1
            elif variant == "B":
                mu = aA * (aB + bB - 1) / (aA + bA) / (aB - 1)
                mup = (aA + 1) * (aB + bB - 2) / (aA + bA + 1) / (aB - 2)
                var = mu * (mup - mu)
                sigma = np.sqrt(var)

                dist = stats.norm(mu, sigma)
                ppfl, ppfu = dist.ppf([lower, upper])

                if method == "asymptotic":
                    return ppfl - 1, ppfu - 1
                else:
                    ppfl = optimize.newton(func=func_ppf, x0=ppfl,
                        args=(aA, bA, aB, bB, lower), maxiter=1000)

                    ppfu = optimize.newton(func=func_ppf, x0=ppfu,
                        args=(aA, bA, aB, bB, upper), maxiter=1000)

                    return ppfl - 1, ppfu - 1
            else:
                return (self.expected_loss_relative_ci(method=method,
                    variant="A", interval_length=interval_length),
                    self.expected_loss_relative_ci(method=method,
                    variant="B", interval_length=interval_length))
