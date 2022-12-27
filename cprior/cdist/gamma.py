"""
Gamma conjugate prior distribution model.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from multiprocessing import Pool

import numpy as np
from scipy import integrate, optimize, special, stats

from .base import BayesABTest, BayesModel, BayesMVTest
from .ci import ci_interval
from .utils import check_ab_method, check_mv_method


def func_ppf(x, a0, b0, a1, b1, p):
    """Function CDF ratio of gamma function for root-finding."""
    return special.betainc(a0, a1, x / (x + b1 / b0)) - p


def func_mv_ppf(x, variant_params, p):
    """Function CDF of max of gamma random variables for root-finding."""
    cdf = 1.0
    for (a, b) in variant_params:
        cdf *= special.gammainc(a, b * x)
    return cdf - p


def func_mv_prob(x, a, b, variant_params):
    """Integrand probability integral."""
    pdf = a * np.log(b) + (a - 1) * np.log(x) - b * x - special.gammaln(a)
    g = np.prod([special.gammainc(a, b * x) for a, b in variant_params],
                axis=0)
    return np.exp(pdf) * g


def func_mv_el(x, a, b, variant_params):
    """Integrand expected loss integral."""
    n = len(variant_params)

    aa, bb = map(np.array, zip(*variant_params))

    pdf = np.exp(aa * np.log(bb) + (aa - 1) * np.log(x) - bb * x
                 - special.gammaln(aa))

    s = np.dot(pdf, [np.prod([special.gammainc(aa[j], bb[j] * x)
               for j in range(n) if j != i], axis=0) for i in range(n)])

    p = x * special.gammainc(a, b * x)
    q = a / b * special.gammainc(a + 1, b * x)
    return s * (p - q)


def func_mv_elr(x, variant_params):
    """Integrand expected loss relative integral."""
    n = len(variant_params)

    aa, bb = map(np.array, zip(*variant_params))

    pdf = np.exp(aa * np.log(bb) + (aa - 1) * np.log(x) - bb * x
                 - special.gammaln(aa))

    s = np.dot(pdf, [np.prod([special.gammainc(aa[j], bb[j] * x)
               for j in range(n) if j != i], axis=0) for i in range(n)])
    return x * s


class GammaModel(BayesModel):
    """
    Gamma conjugate prior distribution model.

    Parameters
    ----------
    name : str (default="")
        Model name.

    shape : int or float
        Prior parameter shape.

    rate : int or float
        Prior parameter rate.
    """
    def __init__(self, name="", shape=0.001, rate=0.001):
        super().__init__(name)

        self.shape = shape
        self.rate = rate

        self._shape_posterior = shape
        self._rate_posterior = rate

        if self.shape <= 0:
            raise ValueError("shape must be > 0; got {}.".format(self.shape))

        if self.rate <= 0:
            raise ValueError("rate must be > 0; got {}.".format(self.rate))

    @property
    def shape_posterior(self):
        """
        Posterior parameter alpha (shape).

        Returns
        -------
        alpha : float
        """
        return self._shape_posterior

    @property
    def rate_posterior(self):
        """
        Posterior parameter beta (rate).

        Returns
        -------
        beta : float
        """
        return self._rate_posterior

    def mean(self):
        """
        Mean of the posterior distribution.

        Returns
        -------
        mean : float
        """
        return stats.gamma(a=self._shape_posterior, loc=0,
                           scale=1.0 / self._rate_posterior).mean()

    def var(self):
        """
        Variance of the posterior distribution.

        Returns
        -------
        var : float
        """
        return stats.gamma(a=self._shape_posterior, loc=0,
                           scale=1.0 / self._rate_posterior).var()

    def std(self):
        """
        Standard deviation of the posterior distribution.

        Returns
        -------
        std : float
        """
        return stats.gamma(a=self._shape_posterior, loc=0,
                           scale=1.0 / self._rate_posterior).std()

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
        return stats.gamma(a=self._shape_posterior, loc=0,
                           scale=1.0 / self._rate_posterior).pdf(x)

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
        return stats.gamma(a=self._shape_posterior, loc=0,
                           scale=1.0 / self._rate_posterior).cdf(x)

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
        return stats.gamma(a=self._shape_posterior, loc=0,
                           scale=1.0 / self._rate_posterior).ppf(q)

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
        return stats.gamma(a=self._shape_posterior, loc=0,
                           scale=1.0 / self._rate_posterior).rvs(
                           size=size, random_state=random_state)


class GammaABTest(BayesABTest):
    """
    Bayesian A/B testing with prior gamma distribution.

    Parameters
    ----------
    modelA : object
        The gamma model for variant A.

    modelB : object
        The gamma model for variant B.

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

        Returns
        -------
        probability : float or tuple of floats
        """
        check_ab_method(method=method, method_options=("exact", "MC"),
                        variant=variant, lift=lift)

        if method == "exact":
            aA = self.modelA.shape_posterior
            bA = self.modelA.rate_posterior

            aB = self.modelB.shape_posterior
            bB = self.modelB.rate_posterior

            if variant == "A":
                p = bB / (bA + bB)
                return special.betainc(aB, aA, p)
            elif variant == "B":
                p = bA / (bA + bB)
                return special.betainc(aA, aB, p)
            else:
                pA = bB / (bA + bB)
                pB = bA / (bA + bB)

                return (special.betainc(aB, aA, pA),
                        special.betainc(aA, aB, pB))
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

        Returns
        -------
        exoected_loss : float or tuple of floats
        """
        check_ab_method(method=method, method_options=("exact", "MC"),
                        variant=variant, lift=lift)

        if method == "exact":
            aA = self.modelA.shape_posterior
            bA = self.modelA.rate_posterior

            aB = self.modelB.shape_posterior
            bB = self.modelB.rate_posterior

            if variant == "A":
                ta = aA / bA * special.betainc(aA + 1, aB, bA / (bA + bB))
                tb = aB / bB * special.betainc(aA, aB + 1, bA / (bA + bB))
                return tb - ta
            elif variant == "B":
                ta = aA / bA * special.betainc(aB, aA + 1, bB / (bA + bB))
                tb = aB / bB * special.betainc(aB + 1, aA, bB / (bA + bB))
                return ta - tb
            else:
                ta = aA / bA * special.betainc(aA + 1, aB, bA / (bA + bB))
                tb = aB / bB * special.betainc(aA, aB + 1, bA / (bA + bB))
                maxba = tb - ta

                ta = aA / bA * special.betainc(aB, aA + 1, bB / (bA + bB))
                tb = aB / bB * special.betainc(aB + 1, aA, bB / (bA + bB))
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
            aA = self.modelA.shape_posterior
            bA = self.modelA.rate_posterior

            aB = self.modelB.shape_posterior
            bB = self.modelB.rate_posterior

            if variant == "A":
                return bA / bB * aB / (aA - 1) - 1
            elif variant == "B":
                return bB / bA * aA / (aB - 1) - 1
            else:
                return (bA / bB * aB / (aA - 1) - 1,
                        bB / bA * aA / (aB - 1) - 1)
        else:
            xA = self.modelA.rvs(self.simulations, self.random_state)
            xB = self.modelB.rvs(self.simulations, self.random_state)

            if variant == "A":
                return ((xB - xA) / xA).mean()
            elif variant == "B":
                return ((xA - xB) / xB).mean()
            else:
                return (((xB - xA) / xA).mean(), ((xA - xB) / xB).mean())

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

            aA = self.modelA.shape_posterior
            bA = self.modelA.rate_posterior

            aB = self.modelB.shape_posterior
            bB = self.modelB.rate_posterior

            mu = aB / bB - aA / bA
            varA = aA / bA ** 2
            varB = aB / bB ** 2
            sigma = np.sqrt(varA + varB)

            if variant == "A":
                return stats.norm(mu, sigma).ppf([lower, upper])
            elif variant == "B":
                return stats.norm(-mu, sigma).ppf([lower, upper])
            else:
                return (stats.norm(mu, sigma).ppf([lower, upper]),
                        stats.norm(-mu, sigma).ppf([lower, upper]))

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
            aA = self.modelA.shape_posterior
            bA = self.modelA.rate_posterior

            aB = self.modelB.shape_posterior
            bB = self.modelB.rate_posterior

            if variant == "A":
                mu = bA / bB * aB / (aA - 1)
                var = aB * (aB + aA - 1) / (aA - 2) / (aA - 1)**2
                var *= (bA / bB) ** 2
                sigma = np.sqrt(var)

                dist = stats.norm(mu, sigma)
                ppfl, ppfu = dist.ppf([lower, upper])

                if method == "asymptotic":
                    return ppfl - 1, ppfu - 1
                else:
                    ppfl = optimize.newton(func=func_ppf, x0=ppfl, args=(
                        aB, bB, aA, bA, lower), maxiter=100)

                    ppfu = optimize.newton(func=func_ppf, x0=ppfu, args=(
                        aB, bB, aA, bA, upper), maxiter=100)

                    return ppfl - 1, ppfu - 1

            elif variant == "B":
                mu = bB / bA * aA / (aB - 1)
                var = aA * (aA + aB - 1) / (aB - 2) / (aB - 1)**2
                var *= (bB / bA) ** 2
                sigma = np.sqrt(var)

                dist = stats.norm(mu, sigma)
                ppfl, ppfu = dist.ppf([lower, upper])

                if method == "asymptotic":
                    return ppfl - 1, ppfu - 1
                else:
                    ppfl = optimize.newton(func=func_ppf, x0=ppfl, args=(
                        aA, bA, aB, bB, lower), maxiter=100)

                    ppfu = optimize.newton(func=func_ppf, x0=ppfu, args=(
                        aA, bA, aB, bB, upper), maxiter=100)

                    return ppfl - 1, ppfu - 1
            else:
                return (self.expected_loss_relative_ci(method=method,
                        variant="A", interval_length=interval_length),
                        self.expected_loss_relative_ci(method=method,
                        variant="B", interval_length=interval_length))


class GammaMVTest(BayesMVTest):
    """
    Bayesian Multivariate testing with prior gamma distribution.

    Parameters
    ----------
    models : object
        The gamma models.

    simulations : int or None (default=1000000)
        Number of Monte Carlo simulations.

    random_state : int or None (default=None)
        The seed used by the random number generator.
    """
    def __init__(self, models, simulations=None, random_state=None,
                 n_jobs=None):
        super().__init__(models, simulations, random_state, n_jobs)

    def probability(self, method="exact", control="A", variant="B", lift=0):
        """
        Compute the error probability or *chance to beat control*, i.e.,
        :math:`P[variant > control + lift]`.

        If ``lift`` is positive value, the computation method must be Monte
        Carlo sampling.

        Parameters
        ----------
        method : str (default="exact")
            The method of computation. Options are "exact" and "MC".

        control : str (default="A")
            The control variant.

        variant : str (default="B")
            The tested variant.

        lift : float (default=0.0)
           The amount of uplift.

        Returns
        -------
        probability : float
        """
        check_mv_method(method=method, method_options=("exact", "MC"),
                        control=control, variant=variant,
                        variants=self.models.keys(), lift=lift)

        model_control = self.models[control]
        model_variant = self.models[variant]

        if method == "exact":
            a0 = model_control.shape_posterior
            b0 = model_control.rate_posterior

            a1 = model_variant.shape_posterior
            b1 = model_variant.rate_posterior

            p = b0 / (b0 + b1)
            return special.betainc(a0, a1, p)
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
            variant_params = [(self.models[v].shape_posterior,
                              self.models[v].rate_posterior) for v in variants]

            a = self.models[variant].shape_posterior
            b = self.models[variant].rate_posterior

            n = self.models[variant].ppf(0.99999999)
            return integrate.quad(func=func_mv_prob, a=0, b=n, args=(
                a, b, variant_params))[0]
        else:
            # prepare parameters
            variant_params = [(self.models[v].shape_posterior,
                              self.models[v].rate_posterior) for v in variants]

            r = np.arange(1, mlhs_samples + 1)
            np.random.shuffle(r)
            v = (r - 0.5) / mlhs_samples
            x = self.models[variant].ppf(v)

            return np.nanmean(np.prod([special.gammainc(a, b * x)
                              for a, b in variant_params], axis=0))

    def expected_loss(self, method="exact", control="A", variant="B", lift=0):
        r"""
        Compute the expected loss. This is the expected uplift lost by choosing
        a given variant, i.e., :math:`\mathrm{E}[\max(control - variant -
        lift, 0)]`.

        If ``lift`` is positive value, the computation method must be Monte
        Carlo sampling.

        Parameters
        ----------
        method : str (default="exact")
            The method of computation. Options are "exact" and "MC".

        control : str (default="A")
            The control variant.

        variant : str (default="B")
            The tested variant.

        lift : float (default=0.0)
           The amount of uplift.

        Returns
        -------
        expected_loss : float
        """
        check_mv_method(method=method, method_options=("exact", "MC"),
                        control=control, variant=variant,
                        variants=self.models.keys(), lift=lift)

        model_control = self.models[control]
        model_variant = self.models[variant]

        if method == "exact":
            a0 = model_control.shape_posterior
            b0 = model_control.rate_posterior

            a1 = model_variant.shape_posterior
            b1 = model_variant.rate_posterior

            t0 = a0 / b0 * special.betainc(a1, a0 + 1, b1 / (b0 + b1))
            t1 = a1 / b1 * special.betainc(a1 + 1, a0, b1 / (b0 + b1))
            return t0 - t1
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

            a0 = model_control.shape_posterior
            b0 = model_control.rate_posterior

            a1 = model_variant.shape_posterior
            b1 = model_variant.rate_posterior

            mu = a1 / b1 - a0 / b0
            var0 = a0 / b0 ** 2
            var1 = a1 / b1 ** 2
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
            a0 = model_control.shape_posterior
            b0 = model_control.rate_posterior

            a1 = model_variant.shape_posterior
            b1 = model_variant.rate_posterior

            return b1 / b0 * a0 / (a1 - 1) - 1
        else:
            x0 = model_control.rvs(self.simulations, self.random_state)
            x1 = model_variant.rvs(self.simulations, self.random_state)

            return ((x0 - x1) / x1).mean()

    def expected_lift_relative(self, method="exact", control="A", variant="B"):
        r"""
        Compute expected relative loss for choosing a variant. This can be seen
        as the negative expected relative improvement or uplift, i.e.,
        :math:`\mathrm{E}[(variant - control) / control]`.

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
            a0 = model_control.shape_posterior
            b0 = model_control.rate_posterior

            a1 = model_variant.shape_posterior
            b1 = model_variant.rate_posterior

            return b0 / b1 * (a1 - 1) / a0 - 1
        else:
            x0 = model_control.rvs(self.simulations, self.random_state)
            x1 = model_variant.rvs(self.simulations, self.random_state)

            return ((x1 - x0) / x0).mean()

    def expected_loss_relative_vs_all(self, method="MLHS", control="A",
                                      variant="B", mlhs_samples=1000):
        r"""
        Compute the expected relative loss against all variations. For example,
        given variants "A", "B", "C" and "D", and choosing variant="B",
        we compute :math:`\mathrm{E}[(\max(A, C, D) - B) / B]`.

        Parameters
        ----------
        method : str (default="MLHS")
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

        variants = list(self.models.keys())

        if method == "MC":
            # exclude variant
            variants.remove(variant)
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
                n = np.max([self.models[v].ppf(0.99999999) for v in variants])

                # exclude variant
                variants.remove(variant)

                # prepare parameters
                variant_params = [(self.models[v].shape_posterior,
                                  self.models[v].rate_posterior)
                                  for v in variants]

                e_max = integrate.quad(func=func_mv_elr, a=0, b=n, args=(
                    variant_params))[0]
            else:
                # exclude variant
                variants.remove(variant)

                e_max = self._expected_value_max_mlhs(variants, mlhs_samples)

            a = self.models[variant].shape_posterior
            b = self.models[variant].rate_posterior
            e_inv_x = b / (a - 1)

            return e_max * e_inv_x - 1

    def expected_lift_relative_vs_all(self, method="MLHS", control="A",
                                      variant="B", mlhs_samples=1000):
        r"""
        Compute the expected relative loss against all variations. For example,
        given variants "A", "B", "C" and "D", and choosing variant="B",
        we compute :math:`\mathrm{E}[(\max(A, C, D) - B) / B]`.

        Parameters
        ----------
        method : str (default="MLHS")
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

        variants = list(self.models.keys())

        if method == "MC":
            # exclude variant
            variants.remove(variant)
            # generate samples from all models in parallel
            xvariant = self.models[variant].rvs(self.simulations,
                                                self.random_state)

            pool = Pool(processes=self.n_jobs)
            processes = [pool.apply_async(self._rvs, args=(v, ))
                         for v in variants]
            xall = [p.get() for p in processes]
            maxall = np.maximum.reduce(xall)

            return (xvariant / maxall).mean() - 1
        else:
            if method == "quad":
                n = np.max([self.models[v].ppf(0.99999999) for v in variants])

                # exclude variant
                variants.remove(variant)

                # prepare parameters
                variant_params = [(self.models[v].shape_posterior,
                                  self.models[v].rate_posterior)
                                  for v in variants]

                e_max = integrate.quad(func=func_mv_elr, a=0, b=n, args=(
                    variant_params))[0]
            else:
                # exclude variant
                variants.remove(variant)

                e_max = self._expected_value_max_mlhs(variants, mlhs_samples)

            a = self.models[variant].shape_posterior
            b = self.models[variant].rate_posterior
            e_x = (a - 1) / b

            return e_x / e_max - 1

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

            a0 = model_control.shape_posterior
            b0 = model_control.rate_posterior

            a1 = model_variant.shape_posterior
            b1 = model_variant.rate_posterior

            mu = b1 / b0 * a0 / (a1 - 1)
            var = a0 * (a0 + a1 - 1) / (a1 - 2) / (a1 - 1)**2
            var *= (b1 / b0) ** 2
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

        variants = list(self.models.keys())

        if method == "MC":
            # exclude variant
            variants.remove(variant)

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
            n = np.max([self.models[v].ppf(0.99999999) for v in variants])

            # exclude variant
            variants.remove(variant)

            # prepare parameters
            variant_params = [(self.models[v].shape_posterior,
                              self.models[v].rate_posterior) for v in variants]

            a = self.models[variant].shape_posterior
            b = self.models[variant].rate_posterior

            if method == "quad":
                return integrate.quad(func=func_mv_el, a=0, b=n, args=(
                    a, b, variant_params))[0]
            else:
                r = np.arange(1, mlhs_samples + 1)
                np.random.shuffle(r)
                v = (r - 0.5) / mlhs_samples

                # ppf of distribution of max(x0, x1, ..., xn), where x_i
                # follows a gamma distribution
                x = np.array([optimize.brentq(f=func_mv_ppf,
                             args=(variant_params, p), a=0, b=n, xtol=1e-4,
                             rtol=1e-4) for p in v])

                p = x * special.gammainc(a, b * x)
                q = a / b * special.gammainc(a + 1, b * x)
                return np.mean(p - q)

    def _expected_value_max_mlhs(self, variants, mlhs_samples):
        """Compute expected value of the maximum of beta random variables."""
        r = np.arange(1, mlhs_samples + 1)
        np.random.shuffle(r)
        v = (r - 0.5) / mlhs_samples
        v = v[..., np.newaxis]

        variant_params = [(self.models[v].shape_posterior,
                          self.models[v].rate_posterior)
                          for v in variants]

        n = len(variant_params)
        aa, bb = map(np.array, zip(*variant_params))
        cc = aa / bb

        xx = stats.gamma(a=aa + 1, loc=0, scale=1.0 / bb).ppf(v)

        return np.sum([cc[i] * np.prod([
                      special.gammainc(aa[j], bb[j] * xx[:, i])
                      for j in range(n) if j != i], axis=0)
                      for i in range(n)], axis=0).mean()
