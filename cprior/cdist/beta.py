"""
Beta conjugate prior distribution model.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from multiprocessing import Pool

import mpmath as mp
import numpy as np
from scipy import integrate, optimize, special, stats

from .._lib.cprior import beta_cprior
from .base import BayesABTest, BayesModel, BayesMVTest
from .ci import ci_interval
from .utils import check_ab_method, check_mv_method


def func_ppf(x, a0, b0, a1, b1, p):
    """Function CDF ratio of beta function for root-finding."""
    mp.mp.dps = 100
    one = mp.mp.one

    c = mp.beta(a0 + a1, b0) / (mp.beta(a0, b0) * mp.beta(a1, b1))
    c *= mp.mpf(x) ** -a1 / a1
    f = mp.hyp3f2(a1, a0 + a1, one - b1, a1 + one, a0 + a1 + b0, one / x)
    return float(one - c * f) - p


def func_mv_ppf(x, variant_params, p):
    """Function CDF of max of beta random variables for root-finding."""
    cdf = 1.0
    for (a, b) in variant_params:
        cdf *= special.betainc(a, b, x)
    return cdf - p


def func_mv_prob(x, a, b, variant_params):
    """Integrand probability integral."""
    pdf = (a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - special.betaln(a, b)
    g = np.prod([special.betainc(a, b, x) for a, b in variant_params], axis=0)
    return np.exp(pdf) * g


def func_mv_el(x, a, b, variant_params):
    """Integrand expected loss integral."""
    n = len(variant_params)

    aa, bb = map(np.array, zip(*variant_params))

    pdf = np.exp((aa - 1) * np.log(x) + (bb - 1) * np.log(1 - x)
                 - special.betaln(aa, bb))

    s = np.dot(pdf, [np.prod([special.betainc(aa[j], bb[j], x)
               for j in range(n) if j != i], axis=0) for i in range(n)])

    p = x * special.betainc(a, b, x)
    q = a / (a + b) * special.betainc(a + 1, b, x)
    return s * (p - q)


def func_mv_elr(x, variant_params):
    """Integrand expected loss relative integral."""
    n = len(variant_params)

    aa, bb = map(np.array, zip(*variant_params))

    pdf = np.exp((aa - 1) * np.log(x) + (bb - 1) * np.log(1 - x)
                 - special.betaln(aa, bb))

    s = np.dot(pdf, [np.prod([special.betainc(aa[j], bb[j], x)
               for j in range(n) if j != i], axis=0) for i in range(n)])
    return x * s


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
    def __init__(self, name="", alpha=1, beta=1):
        super().__init__(name)

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
        """
        Mean of the posterior distribution.

        Returns
        -------
        mean : float
        """
        return stats.beta(self._alpha_posterior, self._beta_posterior).mean()

    def var(self):
        """
        Variance of the posterior distribution.

        Returns
        -------
        var : float
        """
        return stats.beta(self._alpha_posterior, self._beta_posterior).var()

    def std(self):
        """
        Standard deviation of the posterior distribution.

        Returns
        -------
        std : float
        """
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
        q : array-like
            Lower tail probability.

        Returns
        -------
        ppf : numpy.ndarray
            Quantile corresponding to the lower tail probability q.
        """
        return stats.beta(self._alpha_posterior, self._beta_posterior).ppf(q)

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
        return stats.beta(self._alpha_posterior, self._beta_posterior).rvs(
            size=size, random_state=random_state)


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

    def probability(self, method="exact", variant="A", lift=0,
                    mlhs_samples=10000):
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
            The method of computation. Options are "exact", "MC" (Monte Carlo)
            and "MLHS" (Monte Carlo + Median Latin Hypercube Sampling).

        variant : str (default="A")
            The chosen variant. Options are "A", "B", "all".

        lift : float (default=0.0)
           The amount of uplift.

        mlhs_samples : int (default=10000)
            Number of samples for MLHS method.

        Returns
        -------
        probability : float or tuple of floats
        """
        check_ab_method(method=method, method_options=("exact", "MC", "MLHS"),
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
        elif method == "MLHS":
            aA = self.modelA.alpha_posterior
            bA = self.modelA.beta_posterior

            aB = self.modelB.alpha_posterior
            bB = self.modelB.beta_posterior

            r = np.arange(1, mlhs_samples + 1)
            np.random.shuffle(r)
            v = (r - 0.5) / mlhs_samples

            if variant == "A":
                p = np.mean(special.betainc(aA, bA, self.modelB.ppf(v)))
                return 1.0 - p
            elif variant == "B":
                p = np.mean(special.betainc(aB, bB, self.modelA.ppf(v)))
                return 1.0 - p
            else:
                pa = np.mean(special.betainc(aA, bA, self.modelB.ppf(v)))
                pb = np.mean(special.betainc(aB, bB, self.modelA.ppf(v)))
                return 1.0 - pa, 1.0 - pb
        else:
            xA = self.modelA.rvs(self.simulations, self.random_state)
            xB = self.modelB.rvs(self.simulations, self.random_state)

            if variant == "A":
                return (xA > xB + lift).mean()
            elif variant == "B":
                return (xB > xA + lift).mean()
            else:
                return (xA > xB + lift).mean(), (xB > xA + lift).mean()

    def expected_loss(self, method="exact", variant="A", lift=0,
                      mlhs_samples=10000):
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
            The method of computation. Options are "exact", "MC" (Monte Carlo)
            and "MLHS" (Monte Carlo + Median Latin Hypercube Sampling).

        variant : str (default="A")
            The chosen variant. Options are "A", "B", "all".

        lift : float (default=0.0)
            The amount of uplift.

        mlhs_samples : int (default=10000)
            Number of samples for MLHS method.

        Returns
        -------
        expected_loss : float or tuple of floats
        """
        check_ab_method(method=method, method_options=("exact", "MC", "MLHS"),
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
        elif method == "MLHS":
            aA = self.modelA.alpha_posterior
            bA = self.modelA.beta_posterior

            aB = self.modelB.alpha_posterior
            bB = self.modelB.beta_posterior

            r = np.arange(1, mlhs_samples + 1)
            np.random.shuffle(r)
            v = (r - 0.5) / mlhs_samples

            if variant == "A":
                x = self.modelB.ppf(v)
                p = x * special.betainc(aA, bA, x)
                q = aA / (aA + bA) * special.betainc(aA + 1, bA, x)
                return np.mean(p - q)
            elif variant == "B":
                x = self.modelA.ppf(v)
                p = x * special.betainc(aB, bB, x)
                q = aB / (aB + bB) * special.betainc(aB + 1, bB, x)
                return np.mean(p - q)
            else:
                x = self.modelB.ppf(v)
                p = x * special.betainc(aA, bA, x)
                q = aA / (aA + bA) * special.betainc(aA + 1, bA, x)
                pa = np.mean(p - q)

                x = self.modelA.ppf(v)
                p = x * special.betainc(aB, bB, x)
                q = aB / (aB + bB) * special.betainc(aB + 1, bB, x)
                pb = np.mean(p - q)
                return pa, pb
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

    def expected_loss_ci(self, method="MC", variant="A", interval_length=0.9,
                         ci_method="ETI"):
        r"""
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
            Compute ``interval_length`` \% credible interval. This is a value
            in [0, 1].

        ci_method : str (default="ETI")
            Method to compute credible intervals. Supported methods are Highest
            Density interval (``method="HDI``) and Equal-tailed interval
            (``method="ETI"``). Currently, ``method="HDI`` is only available
            for ``method="MC"``.

        Returns
        -------
        expected_loss_ci : np.ndarray or tuple of np.ndarray
        """
        check_ab_method(method=method, method_options=("MC", "asymptotic"),
                        variant=variant, interval_length=interval_length)

        if method == "MC":
            xA = self.modelA.rvs(self.simulations, self.random_state)
            xB = self.modelB.rvs(self.simulations, self.random_state)

            if variant == "A":
                return ci_interval((xB - xA), interval_length, ci_method)
            elif variant == "B":
                return ci_interval((xA - xB), interval_length, ci_method)
            else:
                return (ci_interval((xB - xA), interval_length, ci_method),
                        ci_interval((xA - xB), interval_length, ci_method))
        else:
            lower = (1 - interval_length) / 2
            upper = (1 + interval_length) / 2

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

        Returns
        -------
        expected_loss_relative : float or tuple of floats
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

    def expected_loss_relative_ci(self, method="MC", variant="A",
                                  interval_length=0.9, ci_method="ETI"):
        r"""
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

        ci_method : str (default="ETI")
            Method to compute credible intervals. Supported methods are Highest
            Density interval (``method="HDI``) and Equal-tailed interval
            (``method="ETI"``). Currently, ``method="HDI`` is only available
            for ``method="MC"``.

        Returns
        -------
        expected_loss_relative_ci : np.ndarray or tuple of np.ndarray
        """
        check_ab_method(method=method,
                        method_options=("asymptotic", "exact", "MC"),
                        variant=variant, interval_length=interval_length)

        if method == "MC":
            xA = self.modelA.rvs(self.simulations, self.random_state)
            xB = self.modelB.rvs(self.simulations, self.random_state)

            if variant == "A":
                return ci_interval((xB - xA)/xA, interval_length, ci_method)
            elif variant == "B":
                return ci_interval((xA - xB)/xB, interval_length, ci_method)
            else:
                return (ci_interval((xB - xA)/xA, interval_length, ci_method),
                        ci_interval((xA - xB)/xB, interval_length, ci_method))
        else:
            lower = (1 - interval_length) / 2
            upper = (1 + interval_length) / 2

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
                    ppfl = optimize.newton(func=func_ppf, x0=ppfl, args=(
                        float(aB), float(bB), float(aA), float(bA), lower),
                        maxiter=100)

                    ppfu = optimize.newton(func=func_ppf, x0=ppfu, args=(
                        float(aB), float(bB), float(aA), float(bA), upper),
                        maxiter=100)

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
                    ppfl = optimize.newton(func=func_ppf, x0=ppfl, args=(
                        float(aA), float(bA), float(aB), float(bB), lower),
                        maxiter=1000)

                    ppfu = optimize.newton(func=func_ppf, x0=ppfu, args=(
                        float(aA), float(bA), float(aB), float(bB), upper),
                        maxiter=1000)

                    return ppfl - 1, ppfu - 1
            else:
                return (self.expected_loss_relative_ci(method=method,
                        variant="A", interval_length=interval_length),
                        self.expected_loss_relative_ci(method=method,
                        variant="B", interval_length=interval_length))


class BetaMVTest(BayesMVTest):
    """
    Bayesian Multivariate testing with prior beta distribution.

    Parameters
    ----------
    models : object
        The beta models.

    simulations : int or None (default=1000000)
        Number of Monte Carlo simulations.

    random_state : int or None (default=None)
        The seed used by the random number generator.
    """
    def __init__(self, models, simulations=None, random_state=None,
                 n_jobs=None):
        super().__init__(models, simulations, random_state, n_jobs)

    def probability(self, method="exact", control="A", variant="B", lift=0,
                    mlhs_samples=10000):
        """
        Compute the error probability or *chance to beat control*, i.e.,
        :math:`P[variant > control + lift]`.

        If ``lift`` is positive value, the computation method must be Monte
        Carlo sampling.

        Parameters
        ----------
        method : str (default="exact")
            The method of computation. Options are "exact", "MC" (Monte Carlo)
            and "MLHS" (Monte Carlo + Median Latin Hypercube Sampling).

        control : str (default="A")
            The control variant.

        variant : str (default="B")
            The tested variant.

        lift : float (default=0.0)
           The amount of uplift.

        mlhs_samples : int (default=10000)
            Number of samples for MLHS method.

        Returns
        -------
        probability : float
        """
        check_mv_method(method=method, method_options=("exact", "MC", "MLHS"),
                        control=control, variant=variant,
                        variants=self.models.keys(), lift=lift)

        model_control = self.models[control]
        model_variant = self.models[variant]

        if method == "exact":
            a0 = model_control.alpha_posterior
            b0 = model_control.beta_posterior

            a1 = model_variant.alpha_posterior
            b1 = model_variant.beta_posterior

            return beta_cprior(a0, b0, a1, b1)
        elif method == "MLHS":
            a0 = model_control.alpha_posterior
            b0 = model_control.beta_posterior

            r = np.arange(1, mlhs_samples + 1)
            np.random.shuffle(r)
            v = (r - 0.5) / mlhs_samples

            return np.mean(special.betainc(a0, b0, model_variant.ppf(v)))
        else:
            x0 = model_control.rvs(self.simulations, self.random_state)
            x1 = model_variant.rvs(self.simulations, self.random_state)

            return (x1 > x0 + lift).mean()

    def probability_vs_all(self, method="quad", variant="B", lift=0,
                           mlhs_samples=1000):
        r"""
        Compute the error probability or *chance to beat all* variations. For
        example, given variants "A", "B", "C" and "D", and choosing
        variant="B", we compute :math:`P[B > \max(A, C, D) + lift]`.

        If ``lift`` is positive value, the computation method must be Monte
        Carlo sampling.

        Parameters
        ----------
        method : str (default="quad")
            The method of computation. Options are "MC" (Monte Carlo),
            "MLHS" (Monte Carlo + Median Latin Hypercube Sampling) and "quad"
            (numerical integration).

        variant : str (default="B")
            The chosen variant.

        lift : float (default=0.0)
           The amount of uplift.

        mlhs_samples : int (default=1000)
            Number of samples for MLHS method.

        Returns
        -------
        probability_vs_all : float
        """
        check_mv_method(method=method, method_options=("MC", "MLHS", "quad"),
                        control=None, variant=variant,
                        variants=self.models.keys(), lift=lift)

        # exclude variant
        variants = list(self.models.keys())
        variants.remove(variant)

        if method == "MC":
            # generate samples from all models in parallel
            xvariant = self.models[variant].rvs(self.simulations,
                                                self.random_state)

            pool = Pool(processes=self.n_jobs)
            processes = [pool.apply_async(self._rvs, args=(v, ))
                         for v in variants]
            xall = [p.get() for p in processes]
            maxall = np.maximum.reduce(xall)

            return (xvariant > maxall + lift).mean()
        elif method == "quad":
            # prepare parameters
            variant_params = [(self.models[v].alpha_posterior,
                              self.models[v].beta_posterior) for v in variants]

            a = self.models[variant].alpha_posterior
            b = self.models[variant].beta_posterior

            return integrate.quad(func=func_mv_prob, a=0, b=1, args=(
                a, b, variant_params))[0]
        else:
            # prepare parameters
            variant_params = [(self.models[v].alpha_posterior,
                              self.models[v].beta_posterior) for v in variants]

            r = np.arange(1, mlhs_samples + 1)
            np.random.shuffle(r)
            v = (r - 0.5) / mlhs_samples
            x = self.models[variant].ppf(v)

            return np.mean(np.prod([special.betainc(a, b, x)
                           for a, b in variant_params], axis=0))

    def expected_loss(self, method="exact", control="A", variant="B", lift=0,
                      mlhs_samples=10000):
        r"""
        Compute the expected loss. This is the expected uplift lost by choosing
        a given variant, i.e., :math:`\mathrm{E}[\max(control - variant -
        lift, 0)]`.

        If ``lift`` is positive value, the computation method must be Monte
        Carlo sampling.

        Parameters
        ----------
        method : str (default="exact")
            The method of computation. Options are "exact", "MC" (Monte Carlo)
            and "MLHS" (Monte Carlo + Median Latin Hypercube Sampling).

        control : str (default="A")
            The control variant.

        variant : str (default="B")
            The tested variant.

        lift : float (default=0.0)
           The amount of uplift.

        mlhs_samples : int (default=10000)
            Number of samples for MLHS method.

        Returns
        -------
        expected_loss : float
        """
        check_mv_method(method=method, method_options=("exact", "MC", "MLHS"),
                        control=control, variant=variant,
                        variants=self.models.keys(), lift=lift)

        model_control = self.models[control]
        model_variant = self.models[variant]

        if method == "exact":
            a0 = model_control.alpha_posterior
            b0 = model_control.beta_posterior

            a1 = model_variant.alpha_posterior
            b1 = model_variant.beta_posterior

            t0 = a0 / (a0 + b0) * beta_cprior(a1, b1, a0 + 1, b0)
            t1 = a1 / (a1 + b1) * beta_cprior(a1 + 1, b1, a0, b0)

            return t0 - t1
        elif method == "MLHS":
            a0 = model_control.alpha_posterior
            b0 = model_control.beta_posterior

            a1 = model_variant.alpha_posterior
            b1 = model_variant.beta_posterior

            r = np.arange(1, mlhs_samples + 1)
            np.random.shuffle(r)
            v = (r - 0.5) / mlhs_samples

            x = model_control.ppf(v)
            p = x * special.betainc(a1, b1, x)
            q = a1 / (a1 + b1) * special.betainc(a1 + 1, b1, x)

            return np.mean(p - q)
        else:
            x0 = model_control.rvs(self.simulations, self.random_state)
            x1 = model_variant.rvs(self.simulations, self.random_state)

            return np.maximum(x0 - x1, 0).mean()

    def expected_loss_ci(self, method="MC", control="A", variant="B",
                         interval_length=0.9, ci_method="ETI"):
        r"""
        Compute credible intervals on the difference distribution of
        :math:`Z = control-variant`.

        Parameters
        ----------
        method : str (default="MC")
            The method of computation. Options are "asymptotic" and "MC".

        control : str (default="A")
            The control variant.

        variant : str (default="B")
            The tested variant.

        interval_length : float (default=0.9)
            Compute ``interval_length``\% credible interval. This is a value in
            [0, 1].

        ci_method : str (default="ETI")
            Method to compute credible intervals. Supported methods are Highest
            Density interval (``method="HDI``) and Equal-tailed interval
            (``method="ETI"``). Currently, ``method="HDI`` is only available
            for ``method="MC"``.

        Returns
        -------
        expected_loss_ci : np.ndarray or tuple of np.ndarray
        """
        check_mv_method(method=method, method_options=("MC", "asymptotic"),
                        control=control, variant=variant,
                        variants=self.models.keys(),
                        interval_length=interval_length)

        model_control = self.models[control]
        model_variant = self.models[variant]

        if method == "MC":
            x0 = model_control.rvs(self.simulations, self.random_state)
            x1 = model_variant.rvs(self.simulations, self.random_state)

            return ci_interval((x0 - x1), interval_length, ci_method)
        else:
            lower = (1 - interval_length) / 2
            upper = (1 + interval_length) / 2

            a0 = model_control.alpha_posterior
            b0 = model_control.beta_posterior

            a1 = model_variant.alpha_posterior
            b1 = model_variant.beta_posterior

            mu = a1 / (a1 + b1) - a0 / (a0 + b0)
            var0 = a0 * b0 / (a0 + b0) ** 2 / (a0 + b0 + 1)
            var1 = a1 * b1 / (a1 + b1) ** 2 / (a1 + b1 + 1)
            sigma = np.sqrt(var0 + var1)

            return stats.norm(-mu, sigma).ppf([lower, upper])

    def expected_loss_relative(self, method="exact", control="A", variant="B"):
        r"""
        Compute expected relative loss for choosing a variant. This can be seen
        as the negative expected relative improvement or uplift, i.e.,
        :math:`\mathrm{E}[(control - variant) / variant]`.

        Parameters
        ----------
        method : str (default="exact")
            The method of computation. Options are "exact" and "MC".

        control : str (default="A")
            The control variant.

        variant : str (default="B")
            The tested variant.

        Returns
        -------
        expected_loss_relative : float
        """
        check_mv_method(method=method, method_options=("exact", "MC"),
                        control=control, variant=variant,
                        variants=self.models.keys())

        model_control = self.models[control]
        model_variant = self.models[variant]

        if method == "exact":
            a0 = model_control.alpha_posterior
            b0 = model_control.beta_posterior

            a1 = model_variant.alpha_posterior
            b1 = model_variant.beta_posterior

            return a0 * (a1 + b1 - 1) / (a0 + b0) / (a1 - 1) - 1
        else:
            x0 = model_control.rvs(self.simulations, self.random_state)
            x1 = model_variant.rvs(self.simulations, self.random_state)

            return ((x0 - x1) / x1).mean()

    def expected_lift_relative(self, method="exact", control="A", variant="B"):
        # TODO: docs
        check_mv_method(method=method, method_options=("exact", "MC"),
                        control=control, variant=variant,
                        variants=self.models.keys())

        model_control = self.models[control]
        model_variant = self.models[variant]

        if method == "exact":
            a0 = model_control.alpha_posterior
            b0 = model_control.beta_posterior

            a1 = model_variant.alpha_posterior
            b1 = model_variant.beta_posterior

            return (a0 + b0) * (a1 - 1) / a0 / (a1 + b1 - 1) - 1
        else:
            x0 = model_control.rvs(self.simulations, self.random_state)
            x1 = model_variant.rvs(self.simulations, self.random_state)

            return ((x1 - x0) / x0).mean()

    def expected_loss_relative_vs_all(self, method="quad", control="A",
                                      variant="B", mlhs_samples=1000):
        r"""
        Compute the expected relative loss against all variations. For example,
        given variants "A", "B", "C" and "D", and choosing variant="B",
        we compute :math:`\mathrm{E}[(\max(A, C, D) - B) / B]`.

        Parameters
        ----------
        method : str (default="quad")
            The method of computation. Options are "MC" (Monte Carlo),
            "MLHS" (Monte Carlo + Median Latin Hypercube Sampling) and "quad"
            (numerical integration).

        variant : str (default="B")
            The chosen variant.

        mlhs_samples : int (default=1000)
            Number of samples for MLHS method.

        Returns
        -------
        expected_loss_relative_vs_all : float
        """
        check_mv_method(method=method, method_options=("MC", "MLHS", "quad"),
                        control=None, variant=variant,
                        variants=self.models.keys())

        # exclude variant
        variants = list(self.models.keys())
        variants.remove(variant)

        if method == "MC":
            # generate samples from all models in parallel
            xvariant = self.models[variant].rvs(self.simulations,
                                                self.random_state)

            pool = Pool(processes=self.n_jobs)
            processes = [pool.apply_async(self._rvs, args=(v, ))
                         for v in variants]
            xall = [p.get() for p in processes]
            maxall = np.maximum.reduce(xall)

            return (maxall / xvariant).mean() - 1
        else:
            if method == "quad":
                variant_params = [(self.models[v].alpha_posterior,
                                  self.models[v].beta_posterior)
                                  for v in variants]

                e_max = integrate.quad(func=func_mv_elr, a=0, b=1, args=(
                    variant_params))[0]
            else:
                e_max = self._expected_value_max_mlhs(variants, mlhs_samples)

            a = self.models[variant].alpha_posterior
            b = self.models[variant].beta_posterior
            e_inv_x = (a + b - 1) / (a - 1)

            return e_max * e_inv_x - 1



    def expected_lift_relative_vs_all(self, method="quad", control="A",
                                      variant="B", mlhs_samples=1000):
        # TODO: docs
        check_mv_method(method=method, method_options=("MC", "MLHS", "quad"),
                        control=None, variant=variant,
                        variants=self.models.keys())

        # exclude variant
        variants = list(self.models.keys())
        variants.remove(variant)

        if method == "MC":
            # generate samples from all models in parallel
            xvariant = self.models[variant].rvs(self.simulations,
                                                self.random_state)

            pool = Pool(processes=self.n_jobs)
            processes = [pool.apply_async(self._rvs, args=(v, ))
                         for v in variants]
            xall = [p.get() for p in processes]
            maxall = np.maximum.reduce(xall)

            return (maxall / xvariant).mean() - 1
        else:
            if method == "quad":
                variant_params = [(self.models[v].alpha_posterior,
                                  self.models[v].beta_posterior)
                                  for v in variants]

                e_max = integrate.quad(func=func_mv_elr, a=0, b=1, args=(
                    variant_params))[0]
            else:
                e_max = self._expected_value_max_mlhs(variants, mlhs_samples)

            a = self.models[variant].alpha_posterior
            b = self.models[variant].beta_posterior
            e_x = (a - 1) / (a + b - 1)

            return (e_x / e_max) - 1

    def expected_loss_relative_ci(self, method="MC", control="A", variant="B",
                                  interval_length=0.9, ci_method="ETI"):
        r"""
        Compute credible intervals on the relative difference distribution of
        :math:`Z = (control - variant) / variant`.

        Parameters
        ----------
        method : str (default="MC")
            The method of computation. Options are "asymptotic", "exact" and
            "MC".

        control : str (default="A")
            The control variant.

        variant : str (default="B")
            The tested variant.

        interval_length : float (default=0.9)
            Compute ``interval_length``\% credible interval. This is a value in
            [0, 1].

        ci_method : str (default="ETI")
            Method to compute credible intervals. Supported methods are Highest
            Density interval (``method="HDI``) and Equal-tailed interval
            (``method="ETI"``). Currently, ``method="HDI`` is only available
            for ``method="MC"``.

        Returns
        -------
        expected_loss_relative_ci : np.ndarray or tuple of np.ndarray
        """
        check_mv_method(method=method,
                        method_options=("asymptotic", "exact", "MC"),
                        control=control, variant=variant,
                        variants=self.models.keys(),
                        interval_length=interval_length)

        model_control = self.models[control]
        model_variant = self.models[variant]

        if method == "MC":
            x0 = model_control.rvs(self.simulations, self.random_state)
            x1 = model_variant.rvs(self.simulations, self.random_state)

            return ci_interval((x0 - x1) / x1, interval_length, ci_method)
        else:
            lower = (1 - interval_length) / 2
            upper = (1 + interval_length) / 2

            a0 = model_control.alpha_posterior
            b0 = model_control.beta_posterior

            a1 = model_variant.alpha_posterior
            b1 = model_variant.beta_posterior

            mu = a0 * (a1 + b1 - 1) / (a0 + b0) / (a1 - 1)
            mup = (a0 + 1) * (a1 + b1 - 2) / (a0 + b0 + 1) / (a1 - 2)
            var = mu * (mup - mu)
            sigma = np.sqrt(var)

            dist = stats.norm(mu, sigma)
            ppfl, ppfu = dist.ppf([lower, upper])

            if method == "asymptotic":
                return ppfl - 1, ppfu - 1
            else:
                ppfl = optimize.newton(func=func_ppf, x0=ppfl, args=(
                    a0, b0, a1, b1, lower), maxiter=1000)

                ppfu = optimize.newton(func=func_ppf, x0=ppfu, args=(
                    a0, b0, a1, b1, upper), maxiter=1000)

                return ppfl - 1, ppfu - 1

    def expected_loss_vs_all(self, method="quad", variant="B", lift=0,
                             mlhs_samples=1000):
        r"""
        Compute the expected loss against all variations. For example, given
        variants "A", "B", "C" and "D", and choosing variant="B", we compute
        :math:`\mathrm{E}[\max(\max(A, C, D) - B, 0)]`.

        If ``lift`` is positive value, the computation method must be Monte
        Carlo sampling.

        Parameters
        ----------
        method : str (default="quad")
            The method of computation. Options are "MC" (Monte Carlo),
            "MLHS" (Monte Carlo + Median Latin Hypercube Sampling) and "quad"
            (numerical integration).

        variant : str (default="B")
            The chosen variant.

        lift : float (default=0.0)
           The amount of uplift.

        mlhs_samples : int (default=1000)
            Number of samples for MLHS method.

        Returns
        -------
        expected_loss_vs_all : float
        """
        check_mv_method(method=method, method_options=("MC", "MLHS", "quad"),
                        control=None, variant=variant,
                        variants=self.models.keys(), lift=lift)

        # exclude variant
        variants = list(self.models.keys())
        variants.remove(variant)

        if method == "MC":
            # generate samples from all models in parallel
            xvariant = self.models[variant].rvs(self.simulations,
                                                self.random_state)

            pool = Pool(processes=self.n_jobs)
            processes = [pool.apply_async(self._rvs, args=(v, ))
                         for v in variants]
            xall = [p.get() for p in processes]
            maxall = np.maximum.reduce(xall)

            return np.maximum(maxall - xvariant - lift, 0).mean()
        else:
            variant_params = [(self.models[v].alpha_posterior,
                              self.models[v].beta_posterior) for v in variants]

            a = self.models[variant].alpha_posterior
            b = self.models[variant].beta_posterior

            if method == "quad":
                return integrate.quad(func=func_mv_el, a=0, b=1, args=(
                    a, b, variant_params))[0]
            else:
                r = np.arange(1, mlhs_samples + 1)
                np.random.shuffle(r)
                v = (r - 0.5) / mlhs_samples

                # ppf of distribution of max(x0, x1, ..., xn), where x_i
                # follows a beta distribution
                x = np.array([optimize.brentq(f=func_mv_ppf,
                             args=(variant_params, p), a=0, b=1, xtol=1e-4,
                             rtol=1e-4) for p in v])

                p = x * special.betainc(a, b, x)
                q = a / (a + b) * special.betainc(a + 1, b, x)
                return np.mean(p - q)

    def _expected_value_max_mlhs(self, variants, mlhs_samples):
        """Compute expected value of the maximum of beta random variables."""
        r = np.arange(1, mlhs_samples + 1)
        np.random.shuffle(r)
        v = (r - 0.5) / mlhs_samples
        v = v[..., np.newaxis]

        variant_params = [(self.models[v].alpha_posterior,
                          self.models[v].beta_posterior)
                          for v in variants]

        n = len(variant_params)
        aa, bb = map(np.array, zip(*variant_params))
        cc = aa / (aa + bb)

        xx = stats.beta(aa + 1, bb).ppf(v)

        return np.sum([cc[i] * np.prod([special.betainc(aa[j], bb[j], xx[:, i])
                      for j in range(n) if j != i], axis=0)
                      for i in range(n)], axis=0).mean()
