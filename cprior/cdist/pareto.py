"""
Pareto conjugate prior distribution model.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from scipy import stats

from .base import BayesABTest
from .base import BayesModel
from .base import BayesMVTest
from .utils import check_ab_method
from .utils import check_mv_method


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
        r"""
        Compute credible intervals on the difference distribution of
        :math:`Z = B-A` and/or :math:`Z = A-B`.

        * If ``variant == "A"``, :math:`Z = B - A`
        * If ``variant == "B"``, :math:`Z = A - B`
        * If ``variant == "all"``, both.

        Parameters
        ----------
        method : str (default="MC")
            The method of computation.

        variant : str (default="A")
            The chosen variant. Options are "A", "B", "all".

        interval_length : float (default=0.9)
            Compute ``interval_length``\% credible interval. This is a value in
            [0, 1].
        """
        check_ab_method(method=method, method_options=("MC"),
            variant=variant, interval_length=interval_length)

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

    def expected_loss_relative_ci(self, method="MC", variant="A",
        interval_length=0.9):
        r"""
        Compute credible intervals on the relative difference distribution of
        :math:`Z = (B-A)/A` and/or :math:`Z = (A-B)/B`.

        * If ``variant == "A"``, :math:`Z = (B-A)/A`
        * If ``variant == "B"``, :math:`Z = (A-B)/B`
        * If ``variant == "all"``, both.

        Parameters
        ----------
        method : str (default="MC")
            The method of computation.

        variant : str (default="A")
            The chosen variant. Options are "A", "B", "all".

        interval_length : float (default=0.9)
            Compute ``interval_length``\% credible interval. This is a value in
            [0, 1].
        """
        check_ab_method(method=method,
            method_options=("MC"), variant=variant,
            interval_length=interval_length)

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


class ParetoMVTest(BayesMVTest):
    """
    Bayesian Multivariate testing with prior Pareto distribution.

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
        """
        check_mv_method(method=method, method_options=("exact", "MC"),
            control=control, variant=variant, variants=self.models.keys(),
            lift=lift)

        model_control = self.models[control]
        model_variant = self.models[variant]

        if method == "exact":
            bA = model_variant.scale_posterior
            aA = model_variant.shape_posterior

            bB = model_control.scale_posterior
            aB = model_control.shape_posterior

            return probability_to_beat(aB, bB, aA, bA)
        else:
            x0 = model_control.rvs(self.simulations, self.random_state)
            x1 = model_variant.rvs(self.simulations, self.random_state)

        return (x1 > x0 + lift).mean()

    def probability_vs_all(self, method="MLHS", variant="B", lift=0,
        mlhs_samples=1000):
        r"""
        Compute the error probability or *chance to beat all* variations. For
        example, given variants "A", "B", "C" and "D", and choosing variant="B",
        we compute :math:`P[B > \max(A, C, D) + lift]`.

        If ``lift`` is positive value, the computation method must be Monte
        Carlo sampling.

        Parameters
        ----------
        method : str (default="MLHS")
            The method of computation. Options are "MC" (Monte Carlo)
            and "MLHS" (Monte Carlo + Median Latin Hypercube Sampling).

        variant : str (default="B")
            The chosen variant.

        lift : float (default=0.0)
           The amount of uplift.

        mlhs_samples : int (default=1000)
            Number of samples for MLHS method.
        """
        check_mv_method(method=method, method_options=("MC", "MLHS"),
            control=None, variant=variant, variants=self.models.keys(),
            lift=lift)

        # exclude variant
        variants = list(self.models.keys())
        variants.remove(variant)

        if method == "MC":
            # generate samples from all models in parallel
            xvariant = self.models[variant].rvs(self.simulations,
                self.random_state)

            xall = [self.models[v].rvs(self.simulations, self.random_state) for
                v in variants]
            maxall = np.maximum.reduce(xall)

            return (xvariant > maxall + lift).mean()
        else:
            r = np.arange(mlhs_samples)
            np.random.shuffle(r)
            v = (r - 0.5) / mlhs_samples
            v = v[v >= 0]
            x = self.models[variant].ppf(v)

            return np.nanmean(np.prod([self.models[v].cdf(x)
                for v in variants], axis=0))

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
        """
        check_mv_method(method=method, method_options=("exact", "MC"),
            control=control, variant=variant, variants=self.models.keys(),
            lift=lift)

        model_control = self.models[control]
        model_variant = self.models[variant]

        if method == "exact":
            bA = model_variant.scale_posterior
            aA = model_variant.shape_posterior

            bB = model_control.scale_posterior
            aB = model_control.shape_posterior

            return expected_loss(aA, bA, aB, bB)
        else:
            x0 = model_control.rvs(self.simulations, self.random_state)
            x1 = model_variant.rvs(self.simulations, self.random_state)

            return np.maximum(x0 - x1, 0).mean()

    def expected_loss_ci(self, method="MC", control="A", variant="B",
        interval_length=0.9):
        r"""
        Compute credible intervals on the difference distribution of
        :math:`Z = control-variant`.

        Parameters
        ----------
        method : str (default="MC")
            The method of computation.

        control : str (default="A")
            The control variant.

        variant : str (default="B")
            The tested variant.

        interval_length : float (default=0.9)
            Compute ``interval_length``\% credible interval. This is a value in
            [0, 1].
        """
        check_mv_method(method=method, method_options=("MC"),
            control=control, variant=variant, variants=self.models.keys(),
            interval_length=interval_length)

        # check interval length
        lower = (1 - interval_length) / 2
        upper = (1 + interval_length) / 2

        model_control = self.models[control]
        model_variant = self.models[variant]

        x0 = model_control.rvs(self.simulations, self.random_state)
        x1 = model_variant.rvs(self.simulations, self.random_state)

        lower *= 100.0
        upper *= 100.0

        return np.percentile((x0 - x1), [lower, upper])

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
        """
        check_mv_method(method=method, method_options=("exact", "MC"),
            control=control, variant=variant, variants=self.models.keys())

        model_control = self.models[control]
        model_variant = self.models[variant]

        if method == "exact":
            bA = model_variant.scale_posterior
            aA = model_variant.shape_posterior

            bB = model_control.scale_posterior
            aB = model_control.shape_posterior

            return aB * bB / (aB - 1) * aA / (bA * (aA + 1)) - 1
        else:
            x0 = model_control.rvs(self.simulations, self.random_state)
            x1 = model_variant.rvs(self.simulations, self.random_state)

            return ((x0 - x1) / x1).mean()

    def expected_loss_relative_vs_all(self, method="MLHS", control="A",
        variant="B", mlhs_samples=1000):
        r"""
        Compute the expected relative loss against all variations. For example,
        given variants "A", "B", "C" and "D", and choosing variant="B",
        we compute :math:`\mathrm{E}[(\max(A, C, D) - B) / B]`.

        Parameters
        ----------
        method : str (default="MLHS")
            The method of computation. Options are "MC" (Monte Carlo)
            and "MLHS" (Monte Carlo + Median Latin Hypercube Sampling).

        variant : str (default="B")
            The chosen variant.

        mlhs_samples : int (default=1000)
            Number of samples for MLHS method.
        """
        check_mv_method(method=method, method_options=("MC", "MLHS"),
            control=None, variant=variant, variants=self.models.keys())

        # exclude variant
        variants = list(self.models.keys())
        variants.remove(variant)

        if method == "MC":
            # generate samples from all models in parallel
            xvariant = self.models[variant].rvs(self.simulations,
                self.random_state)

            xall = [self.models[v].rvs(self.simulations, self.random_state) for
                v in variants]
            maxall = np.maximum.reduce(xall)

            return (maxall / xvariant).mean() - 1
        else:
            e_max = self._expected_value_max_mlhs(variants, mlhs_samples)

            a = self.models[variant].shape_posterior
            b = self.models[variant].scale_posterior
            e_inv_x = a / (b * (a + 1))

            return e_max * e_inv_x - 1

    def expected_loss_relative_ci(self, method="MC", control="A", variant="B",
        interval_length=0.9):
        r"""
        Compute credible intervals on the relative difference distribution of
        :math:`Z = (control - variant) / variant`.

        Parameters
        ----------
        method : str (default="MC")
            The method of computation.

        control : str (default="A")
            The control variant.

        variant : str (default="B")
            The tested variant.

        interval_length : float (default=0.9)
            Compute ``interval_length``\% credible interval. This is a value in
            [0, 1].
        """
        check_mv_method(method=method, method_options=("MC"), control=control,
            variant=variant, variants=self.models.keys(),
            interval_length=interval_length)

        lower = (1 - interval_length) / 2
        upper = (1 + interval_length) / 2

        model_control = self.models[control]
        model_variant = self.models[variant]

        x0 = model_control.rvs(self.simulations, self.random_state)
        x1 = model_variant.rvs(self.simulations, self.random_state)

        lower *= 100.0
        upper *= 100.0

        return np.percentile((x0 - x1) / x1, [lower, upper])

    def expected_loss_vs_all(self, method="MC", variant="B", lift=0):
        r"""
        Compute the expected loss against all variations. For example, given
        variants "A", "B", "C" and "D", and choosing variant="B", we compute
        :math:`\mathrm{E}[\max(\max(A, C, D) - B, 0)]`.

        If ``lift`` is positive value, the computation method must be Monte
        Carlo sampling.

        Parameters
        ----------
        method : str (default="MC")
            The method of computation.

        variant : str (default="B")
            The chosen variant.

        lift : float (default=0.0)
           The amount of uplift.
        """
        check_mv_method(method=method, method_options=("MC"),
            control=None, variant=variant, variants=self.models.keys(),
            lift=lift)

        # exclude variant
        variants = list(self.models.keys())
        variants.remove(variant)

        # generate samples from all models in parallel
        xvariant = self.models[variant].rvs(self.simulations,
            self.random_state)

        xall = [self.models[v].rvs(self.simulations, self.random_state) for
            v in variants]
        maxall = np.maximum.reduce(xall)

        return np.maximum(maxall - xvariant - lift, 0).mean()

    def _expected_value_max_mlhs(self, variants, mlhs_samples):
        """Compute expected value of the maximum of gamma random variables."""
        r = np.arange(mlhs_samples)
        np.random.shuffle(r)
        v = (r - 0.5) / mlhs_samples
        v = v[v >= 0]

        s = 0
        for i in variants:
            a = self.models[i].shape_posterior
            b = self.models[i].scale_posterior
            x = stats.pareto(b=a - 1, scale=b).ppf(v)
            c = a * b / (a - 1)
            s += c * np.prod([self.models[j].cdf(x) for j in variants if j != i
                ], axis=0).mean()

        return s
