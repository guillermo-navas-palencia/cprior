"""
Normal-inverse-gamma prior distribution model.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from multiprocessing import Pool
from scipy import integrate
from scipy import optimize
from scipy import special
from scipy import stats

from .base import BayesABTest
from .base import BayesModel
from .base import BayesMVTest
from .gamma import func_ppf
from .utils import check_ab_method
from .utils import check_mv_method


def func_ab_prob(x, muA, sA, vA, muB, sB, vB):
    tA = (x - muA) / sA
    pdf = np.exp(-0.5 * (1 + vA) * np.log(1 + tA ** 2 / vA) - 0.5 * np.log(vA)
                 - np.log(sA) - special.betaln(vA * 0.5, 0.5))
    cdf = special.stdtr(vB, (x - muB) / sB)
    return pdf * cdf


def func_ab_el(x, muA, sA, vA, muB, sB, vB):
    tA = (x - muA) / sA
    tB = (x - muB) / sB

    pdf = np.exp(-0.5 * (1 + vA) * np.log(1 + tA ** 2 / vA) - 0.5 * np.log(vA)
                 - np.log(sA) - special.betaln(vA*0.5, 0.5))

    c = sB * np.exp(0.5 * (1 - vB) * np.log(1 + tB ** 2 / vB)
                    + 0.5 * np.log(vB)
                    - special.betaln(vB * 0.5, 0.5)) / (1 - vB)
    return ((x - muB) * special.stdtr(vB, tB) - c) * pdf


def func_mv_student_ppf(x, variant_params, p):
    """Function CDF of max of student t random variables for root-finding."""
    cdf = 1.0
    for (mu, la, a, b) in variant_params:
        cdf *= special.stdtr(2 * a, (x - mu) / np.sqrt(b / a / la))
    return cdf - p


def func_mv_inverse_gamma_ppf(x, variant_params, p):
    """
    Function CDF of max of inverse gamma random variables for root-finding.
    """
    cdf = 1.0
    x = np.maximum(x, 1e-15)
    for (_, _, a, b) in variant_params:
        cdf *= special.gammaincc(a, b / x)
    return cdf - p


def func_mv_prob_mean(x, mu, s, v, variant_params):
    """Integratnd probability integral."""
    t = (x - mu) / s
    pdf = np.exp(-0.5 * (1 + v) * np.log(1 + t ** 2 / v)
                 - 0.5 * np.log(v) - np.log(s) - special.betaln(v * 0.5, 0.5))

    cdf = np.prod([special.stdtr(2 * a, (x - mu) / np.sqrt(b / a / la))
                   for (mu, la, a, b) in variant_params], axis=0)
    return pdf * cdf


def func_mv_prob_var(x, a, b, variant_params):
    """Integratnd probability integral."""
    pdf = np.exp(a * np.log(b) - (a + 1) * np.log(x) - b / x
                 - special.gammaln(a))
    cdf = np.prod([special.gammaincc(a, b / x)
                   for (_, _, a, b) in variant_params], axis=0)
    return pdf * cdf


def func_mv_el_mean(x, mu, s, v, variant_params):
    """Integrand expected loss integral."""
    n = len(variant_params)

    uu, ll, aa, bb = map(np.array, zip(*variant_params))

    vv = 2 * aa
    ss = np.sqrt(bb / aa / ll)
    tt = (x - uu) / ss

    pdf = np.exp(-0.5 * (1 + vv) * np.log(1 + tt ** 2 / vv) - 0.5 * np.log(vv)
                 - np.log(ss) - special.betaln(vv * 0.5, 0.5))

    pdf = np.dot(pdf, [np.prod([special.stdtr(vv[j], tt[j])
                       for j in range(n) if j != i], axis=0)
                       for i in range(n)])

    t = (x - mu) / s
    c = s * np.exp(0.5 * (1 - v) * np.log(1 + t ** 2 / v) + 0.5 * np.log(v)
                   - special.betaln(v * 0.5, 0.5)) / (1 - v)

    return ((x - mu) * special.stdtr(v, t) - c) * pdf


def func_mv_el_var(x, a, b, variant_params):
    """Integrand expected loss integral."""
    n = len(variant_params)

    _, _, aa, bb = map(np.array, zip(*variant_params))

    pdf = np.exp(aa * np.log(bb) - (aa + 1) * np.log(x) - bb / x
                 - special.gammaln(aa))

    pdf = np.dot(pdf, [np.prod([special.gammaincc(aa[j], bb[j] / x)
                       for j in range(n) if j != i], axis=0)
                       for i in range(n)])

    p = x * special.gammaincc(a, b / x)
    q = b / (a - 1) * special.gammaincc(a - 1, b / x)
    return (p - q) * pdf


def func_mv_elr_mean(x, variant_params):
    """Integrand expected loss relative integral."""
    n = len(variant_params)

    uu, ll, aa, bb = map(np.array, zip(*variant_params))

    vv = 2 * aa
    ss = np.sqrt(bb / aa / ll)
    tt = (x - uu) / ss

    pdf = np.exp(-0.5 * (1 + vv) * np.log(1 + tt ** 2 / vv) - 0.5 * np.log(vv)
                 - np.log(ss) - special.betaln(vv * 0.5, 0.5))

    pdf = np.dot(pdf, [np.prod([special.stdtr(vv[j], tt[j])
                       for j in range(n) if j != i], axis=0)
                       for i in range(n)])
    return x * pdf


def func_mv_elr_var(x, variant_params):
    """Integrand expected loss relative integral."""
    n = len(variant_params)

    _, _, aa, bb = map(np.array, zip(*variant_params))

    pdf = np.exp(aa * np.log(bb) - (aa + 1) * np.log(x) - bb / x
                 - special.gammaln(aa))

    pdf = np.dot(pdf, [np.prod([special.gammaincc(aa[j], bb[j] / x)
                       for j in range(n) if j != i], axis=0)
                       for i in range(n)])
    return x * pdf


class NormalInverseGamma(object):
    """
    Normal-inverse-gamma distribution.

    Parameters
    ----------
    loc : float, optional (default=0)
        Mean of the distribution.

    variance_scale : float, optional (default=1)
        Scale on the normal distribution prior.

    shape : float, optional (default=1)
        Shape of the distribution.

    scale : float, optional (default=1)
        Scale on the inverse gamma distribution prior.
    """
    def __init__(self, loc=0, variance_scale=1, shape=1, scale=1):
        self.loc = loc
        self.variance_scale = variance_scale
        self.shape = shape
        self.scale = scale

        if self.variance_scale <= 0:
            raise ValueError("variance_scale must be > 0; got {}.".format(
                self.variance_scale))

        if self.shape <= 0:
            raise ValueError("shape must be > 0; got {}.".format(self.shape))

        if self.scale <= 0:
            raise ValueError("scale must be > 0; got {}.".format(self.scale))

    def mean(self):
        """
        Mean of the Normal-inverse-gamma probability.

        Returns
        -------
        (x_mean, sig2_mean) : tuple of means.
            Mean of the random variates.
        """
        x_mean = self.loc

        if self.shape > 1:
            sig2_mean = self.scale / (self.shape - 1)
        else:
            sig2_mean = np.nan

        return x_mean, sig2_mean

    def mode(self):
        """
        Mode of the Normal-inverse-gamma probability.

        Returns
        -------
        (x_mode, sig2_mode) : tuple of modes.
            Mode of the random variates.
        """
        x_mode = self.loc
        sig2_mode = self.scale / (self.shape + 1.5)

        return x_mode, sig2_mode

    def var(self):
        """
        Variance of the Normal-inverse-gamma probability.

        Returns
        -------
        (x_var, sig2_var) : tuple of variances.
            Variance of the random variates.
        """
        if self.shape > 1:
            x_var = self.scale / (self.shape - 1) / self.variance_scale
        else:
            x_var = np.nan

        if self.shape > 2:
            sig2_var = self.scale ** 2 / (self.shape - 1) ** 2
            sig2_var /= (self.shape - 2)
        else:
            sig2_var = np.nan

        return x_var, sig2_var

    def std(self):
        """
        Standard deviation of the Normal-inverse-gamma probability.

        Returns
        -------
        (x_std, sig2_std) : tuple of standard deviations.
            Standard deviation of the random variates.
        """
        return tuple(np.sqrt(self.var()))

    def logpdf(self, x, sig2):
        """
        Log of the Normal-inverse-gamma probability density function.

        Parameters
        ----------
        x: array-like
            Quantiles.

        sig2 : array-like
            Quantiles.

        Returns
        -------
        logpdf : numpy.ndarray
            Log of the probability density function evaluated at (x, sig2).
        """
        x, sig2 = self._check_input(x, sig2)

        logsig2 = np.log(sig2)
        t0 = 0.5 * np.log(self.variance_scale) - 0.9189385332046727
        t1 = self.shape * np.log(self.scale) - special.gammaln(self.shape)
        t2 = -(self.shape + 1.5) * logsig2
        t3 = self.scale + 0.5 * self.variance_scale * (x - self.loc) ** 2

        return t0 + t1 + t2 - t3 / sig2

    def pdf(self, x, sig2):
        """
        Normal-inverse-gamma probability density function.

        Parameters
        ----------
        x: array-like
            Quantiles.

        sig2 : array-like
            Quantiles.

        Returns
        -------
        pdf : numpy.ndarray
            Probability density function evaluated at (x, sig2).
        """
        return np.exp(self.logpdf(x, sig2))

    def logcdf(self, x, sig2):
        """
        Log of the Normal-inverse-gamma cumulative distribution function.

        Parameters
        ----------
        x: array-like
            Quantiles.

        sig2 : array-like
            Quantiles.

        Returns
        -------
        logcdf : numpy.ndarray
            Log of the cumulative distribution function evaluated at (x, sig2).
        """
        x, sig2 = self._check_input(x, sig2)

        xu = (self.variance_scale / sig2) ** 0.5 * (x - self.loc)
        t0 = -self.scale / sig2 + self.shape * np.log(self.scale)
        t1 = -(self.shape + 1) * np.log(sig2) - special.gammaln(self.shape)
        t2 = special.log_ndtr(xu)

        return t0 + t1 + t2

    def cdf(self, x, sig2):
        """
        Normal-inverse-gamma cumulative distribution function.

        Parameters
        ----------
        x: array-like
            Quantiles.

        sig2 : array-like
            Quantiles.

        Returns
        -------
        cdf : numpy.ndarray
            Cumulative distribution function evaluated at (x, sig2).
        """
        return np.exp(self.logcdf(x, sig2))

    def rvs(self, size=1, random_state=None):
        """
        Normal-inverse-gamma random variates.

        Parameters
        ----------
        size : int (default=1)
            Number of random variates.

        random_state : int or None (default=None)
            The seed used by the random number generator.

        Returns
        -------
        rvs : numpy.ndarray
            Random variates of given size (size, 2).
        """
        sig2_rv = stats.invgamma(a=self.shape, scale=self.scale).rvs(
            size=size, random_state=random_state)

        x_rv = stats.norm(loc=self.loc,
                          scale=np.sqrt(sig2_rv / self.variance_scale)).rvs(
                          size=size, random_state=random_state)

        return np.c_[x_rv, sig2_rv]

    def _check_input(self, x, sig2):
        x = np.asarray(x)
        sig2 = np.asarray(sig2)

        x_shape, sig2_shape = x.shape, sig2.shape

        if x_shape != sig2_shape:
            raise ValueError("Input variables with inconsistent dimensions. "
                             "{} != {}".format(x_shape, sig2_shape))

        if np.any(sig2 <= 0):
            raise ValueError("sig2 must be > 0.")

        return x, sig2


class NormalInverseGammaModel(BayesModel):
    """
    Normal-inverse-gamma prior distribution model.

    Parameters
    ----------
    name : str, optional (default="")
        Model name.

    loc : float, optional (default=0.001)
        Prior parameter location.

    variance_scale : float, optional (default=0.001)
        Prior parameter variance scale.

    shape : float, optional (default=0.001)
        Prior parameter shape.

    scale : float, optional (default=0.001)
        Prior parameter scale.
    """
    def __init__(self, name="", loc=0.001, variance_scale=0.001, shape=0.001,
                 scale=0.001):
        super().__init__(name)

        self.loc = loc
        self.variance_scale = variance_scale
        self.shape = shape
        self.scale = scale

        self._loc_posterior = loc
        self._variance_scale_posterior = variance_scale
        self._shape_posterior = shape
        self._scale_posterior = scale

        if self.variance_scale <= 0:
            raise ValueError("variance_scale must be > 0; got {}.".format(
                self.variance_scale))

        if self.shape <= 0:
            raise ValueError("shape must be > 0; got {}.".format(self.shape))

        if self.scale <= 0:
            raise ValueError("scale must be > 0; got {}.".format(self.scale))

    @property
    def loc_posterior(self):
        """
        Posterior parameter mu (location).

        Returns
        -------
        mu : float
        """
        return self._loc_posterior

    @property
    def variance_scale_posterior(self):
        """
        Posterior parameter lambda (variance_scale).

        Returns
        -------
        lambda : float
        """
        return self._variance_scale_posterior

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
    def scale_posterior(self):
        """
        Posterior parameter beta (scale).

        Returns
        -------
        beta : float
        """
        return self._scale_posterior

    def mean(self):
        """
        Mean of the posterior distribution.

        Returns
        -------
        mean : tuple of floats
        """
        return NormalInverseGamma(
            loc=self._loc_posterior,
            variance_scale=self._variance_scale_posterior,
            shape=self._shape_posterior, scale=self._scale_posterior).mean()

    def var(self):
        """
        Variance of the posterior distribution.

        Returns
        -------
        var : tuple of floats
        """
        return NormalInverseGamma(
            loc=self._loc_posterior,
            variance_scale=self._variance_scale_posterior,
            shape=self._shape_posterior, scale=self._scale_posterior).var()

    def std(self):
        """
        Standard deviation of the posterior distribution.

        Returns
        -------
        std : tuple of floats
        """
        return NormalInverseGamma(
            loc=self._loc_posterior,
            variance_scale=self._variance_scale_posterior,
            shape=self._shape_posterior, scale=self._scale_posterior).std()

    def pdf(self, x, sig2):
        """
        Probability density function of the posterior distribution.

        Parameters
        ----------
        x : array-like
            Quantiles.

        sig2 : array-like
            Quantiles.

        Returns
        -------
        pdf : numpy.ndarray
           Probability density function evaluated at (x, sig2).
        """
        return NormalInverseGamma(
            loc=self._loc_posterior,
            variance_scale=self._variance_scale_posterior,
            shape=self._shape_posterior, scale=self._scale_posterior
            ).pdf(x, sig2)

    def cdf(self, x, sig2):
        """
        Cumulative distribution function of the posterior distribution.

        Parameters
        ----------
        x : array-like
            Quantiles.

        sig2 : array-like
            Quantiles.

        Returns
        -------
        cdf : numpy.ndarray
            Cumulative distribution function evaluated at (x, sig2).
        """
        return NormalInverseGamma(
            loc=self._loc_posterior,
            variance_scale=self._variance_scale_posterior,
            shape=self._shape_posterior,
            scale=self._scale_posterior).cdf(x, sig2)

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
        rvs : numpy.ndarray
            Random variates of given size (size, 2).
        """
        return NormalInverseGamma(
            loc=self._loc_posterior,
            variance_scale=self._variance_scale_posterior,
            shape=self._shape_posterior, scale=self._scale_posterior).rvs(
            size=size, random_state=random_state)


class NormalInverseGammaABTest(BayesABTest):
    """
    Bayesian A/B testing with prior normal-inverse-gamma distribution.

    Parameters
    ----------
    modelA : object
        The normal-inverse-gamma model for variant A.

    modelB : object
        The normal-inverse-gamma model for variant B.

    simulations : int or None (default=1000000)
        Number of Monte Carlo simulations.

    random_state : int or None (default=None)
        The seed used by the random number generator.
    """
    def __init__(self, modelA, modelB, simulations=1000000, random_state=None):
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
        probability : tuple of floats

        Notes
        -----
        Method "exact" uses the normal approximation of the Student's
        t-distribution for the error probability of the mean when the number
        of degrees of freedom is large. For small values, numerical
        intergration is used.
        """
        check_ab_method(method=method, method_options=("exact", "MC"),
                        variant=variant, lift=lift)

        if method == "exact":
            muA = self.modelA.loc_posterior
            laA = self.modelA.variance_scale_posterior
            aA = self.modelA.shape_posterior
            bA = self.modelA.scale_posterior

            muB = self.modelB.loc_posterior
            laB = self.modelB.variance_scale_posterior
            aB = self.modelB.shape_posterior
            bB = self.modelB.scale_posterior

            sigA = self.modelA.std()[0]
            sigB = self.modelB.std()[0]

            if variant == "A":
                # mean
                if min(aA, aB) > 50:
                    # mean using normal approximation
                    sigA = self.modelA.std()[0]
                    sigB = self.modelB.std()[0]

                    prob_mean = special.ndtr((muA - muB) /
                                             np.hypot(sigA, sigB))
                else:
                    # numerical integration
                    sA = np.sqrt(bA / aA / laA)
                    sB = np.sqrt(bB / aB / laB)

                    prob_mean = integrate.quad(
                        func=func_ab_prob, a=-np.inf, b=np.inf, args=(
                            muA, sA, 2*aA, muB, sB, 2*aB))[0]

                # variance
                p = bA / (bA + bB)
                prob_var = special.betainc(aA, aB, p)
                return prob_mean, prob_var

            elif variant == "B":
                if min(aA, aB) > 50:
                    # mean using normal approximation
                    sigA = self.modelA.std()[0]
                    sigB = self.modelB.std()[0]

                    prob_mean = special.ndtr((muB - muA) /
                                             np.hypot(sigA, sigB))
                else:
                    # numerical integration
                    sA = np.sqrt(bA / aA / laA)
                    sB = np.sqrt(bB / aB / laB)

                    prob_mean = integrate.quad(
                        func=func_ab_prob, a=-np.inf, b=np.inf, args=(
                            muB, sB, 2*aB, muA, sA, 2*aA))[0]

                # variance
                p = bB / (bA + bB)
                prob_var = special.betainc(aB, aA, p)
                return prob_mean, prob_var
            else:
                prob_mean_A, prob_var_A = self.probability(
                    method=method, variant="A", lift=lift)

                prob_mean_B, prob_var_B = self.probability(
                    method=method, variant="B", lift=lift)

                return (prob_mean_A, prob_var_A), (prob_mean_B, prob_var_B)
        else:
            data_A = self.modelA.rvs(self.simulations, self.random_state)
            data_B = self.modelB.rvs(self.simulations, self.random_state)

            xA, sig2A = data_A[:, 0], data_A[:, 1]
            xB, sig2B = data_B[:, 0], data_B[:, 1]

            if variant == "A":
                return (xA > xB + lift).mean(), (sig2A > sig2B + lift).mean()
            elif variant == "B":
                return (xB > xA + lift).mean(), (sig2B > sig2A + lift).mean()
            else:
                return ((xA > xB + lift).mean(),
                        (sig2A > sig2B + lift).mean()), (
                        (xB > xA + lift).mean(),
                        (sig2B > sig2A + lift).mean())

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
        expected_loss : tuple of floats

        Notes
        -----
        Method "exact" uses the normal approximation of the Student's
        t-distribution for the expected loss of the mean when the number
        of degrees of freedom is large. For small values, numerical
        intergration is used.
        """
        check_ab_method(method=method, method_options=("exact", "MC"),
                        variant=variant, lift=lift)

        if method == "exact":
            muA = self.modelA.loc_posterior
            laA = self.modelA.variance_scale_posterior
            aA = self.modelA.shape_posterior
            bA = self.modelA.scale_posterior

            muB = self.modelB.loc_posterior
            laB = self.modelB.variance_scale_posterior
            aB = self.modelB.shape_posterior
            bB = self.modelB.scale_posterior

            sigA = self.modelA.std()[0]
            sigB = self.modelB.std()[0]

            if min(aA, aB) > 50 or max(aA, aB) <= 1:
                u = muB - muA
                s = np.hypot(sigA, sigB)

                t0 = s * np.exp(-0.5 * (u / s) ** 2) / np.sqrt(2 * np.pi)

            if variant == "A":
                if min(aA, aB) > 50 or max(aA, aB) <= 1:
                    # mean using normal approximation
                    el_mean = t0 + u * special.ndtr(u / s)
                else:
                    # numerical integration
                    sA = np.sqrt(bA / aA / laA)
                    sB = np.sqrt(bB / aB / laB)

                    el_mean = integrate.quad(
                        func=func_ab_el, a=-np.inf, b=np.inf, args=(
                            muB, sB, 2*aB, muA, sA, 2*aA))[0]

                # variance
                if min(aA, aB) > 1:
                    ta = bA / (aA - 1) * special.betainc(
                        aB, aA - 1, bB / (bA + bB))
                    tb = bB / (aB - 1) * special.betainc(
                        aB - 1, aA, bB / (bA + bB))
                    el_var = tb - ta
                else:
                    el_var = np.nan
                return el_mean, el_var
            elif variant == "B":
                if min(aA, aB) > 50 or max(aA, aB) <= 1:
                    # mean using normal approximation
                    el_mean = t0 - u * special.ndtr(-u / s)
                else:
                    # numerical integration
                    sA = np.sqrt(bA / aA / laA)
                    sB = np.sqrt(bB / aB / laB)

                    el_mean = integrate.quad(
                        func=func_ab_el, a=-np.inf, b=np.inf, args=(
                            muA, sA, 2*aA, muB, sB, 2*aB))[0]

                # variance
                if min(aA, aB) > 1:
                    ta = bA / (aA - 1) * special.betainc(
                        aA - 1, aB, bA / (bA + bB))
                    tb = bB / (aB - 1) * special.betainc(
                        aA, aB - 1, bA / (bA + bB))
                    el_var = ta - tb
                else:
                    el_var = np.nan
                return el_mean, el_var
            else:
                el_mean_A, el_var_A = self.expected_loss(
                    method=method, variant="A", lift=lift)

                el_mean_B, el_var_B = self.expected_loss(
                    method=method, variant="B", lift=lift)

                return (el_mean_A, el_var_A), (el_mean_B, el_var_B)
        else:
            data_A = self.modelA.rvs(self.simulations, self.random_state)
            data_B = self.modelB.rvs(self.simulations, self.random_state)

            xA, sig2A = data_A[:, 0], data_A[:, 1]
            xB, sig2B = data_B[:, 0], data_B[:, 1]

            if variant == "A":
                return (np.maximum(xB - xA - lift, 0).mean(),
                        np.maximum(sig2B - sig2A - lift, 0).mean())
            elif variant == "B":
                return (np.maximum(xA - xB - lift, 0).mean(),
                        np.maximum(sig2A - sig2B - lift, 0).mean())
            else:
                return (np.maximum(xB - xA - lift, 0).mean(),
                        np.maximum(sig2B - sig2A - lift, 0).mean()), (
                        np.maximum(xA - xB - lift, 0).mean(),
                        np.maximum(sig2A - sig2B - lift, 0).mean())

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
        expected_loss_relative : tuple of floats

        Notes
        -----
        Method "exact" uses an approximation of :math:`E[1/X]` where
        :math:`X` follows a Student's t-distribution.
        """
        check_ab_method(method=method, method_options=("exact", "MC"),
                        variant=variant)

        if method == "exact":
            muA = self.modelA.loc_posterior
            aA = self.modelA.shape_posterior
            bA = self.modelA.scale_posterior

            muB = self.modelB.loc_posterior
            aB = self.modelB.shape_posterior
            bB = self.modelB.scale_posterior

            sig2A = self.modelA.var()[0]
            sig2B = self.modelB.var()[0]

            if variant == "A":
                # mean using asymptotic normal approximation
                elr_mean = muB / muA + sig2A * muB / muA ** 3 - 1

                # variance
                elr_var = bB / bA * aA / (aB - 1) - 1

                return elr_mean, elr_var
            elif variant == "B":
                # mean using asymptotic normal approximation
                elr_mean = muA / muB + sig2B * muA / muB ** 3 - 1

                # variance
                elr_var = bA / bB * aB / (aA - 1) - 1

                return elr_mean, elr_var
            else:
                elr_mean_ba = muB / muA + sig2A * muB / muA ** 3 - 1
                elr_var_ba = bB / bA * aA / (aB - 1) - 1
                elr_mean_ab = muA / muB + sig2B * muA / muB ** 3 - 1
                elr_var_ab = bA / bB * aB / (aA - 1) - 1

                return (elr_mean_ba, elr_var_ba), (elr_mean_ab, elr_var_ab)
        else:
            data_A = self.modelA.rvs(self.simulations, self.random_state)
            data_B = self.modelB.rvs(self.simulations, self.random_state)

            xA, sig2A = data_A[:, 0], data_A[:, 1]
            xB, sig2B = data_B[:, 0], data_B[:, 1]

            if variant == "A":
                return (((xB - xA) / xA).mean(),
                        ((sig2B - sig2A) / sig2A).mean())
            elif variant == "B":
                return (((xA - xB) / xB).mean(),
                        ((sig2A - sig2B) / sig2B).mean())
            else:
                return (((xB - xA) / xA).mean(),
                        ((sig2B - sig2A) / sig2A).mean()), (
                        ((xA - xB) / xB).mean(),
                        ((sig2A - sig2B) / sig2B).mean())

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
            The method of computation. Options are "asymptotic" and "MC".

        variant : str (default="A")
            The chosen variant. Options are "A", "B", "all".

        interval_length : float (default=0.9)
            Compute ``interval_length``\% credible interval. This is a value in
            [0, 1].

        Returns
        -------
        expected_loss_ci : tuple of floats
        """
        check_ab_method(method=method, method_options=("MC", "asymptotic"),
                        variant=variant, interval_length=interval_length)

        # check interval length
        lower = (1 - interval_length) / 2
        upper = (1 + interval_length) / 2

        if method == "MC":
            data_A = self.modelA.rvs(self.simulations, self.random_state)
            data_B = self.modelB.rvs(self.simulations, self.random_state)

            xA, sig2A = data_A[:, 0], data_A[:, 1]
            xB, sig2B = data_B[:, 0], data_B[:, 1]

            lower *= 100.0
            upper *= 100.0

            if variant == "A":
                return (np.percentile((xB - xA), [lower, upper]),
                        np.percentile((sig2B - sig2A), [lower, upper]))
            elif variant == "B":
                return (np.percentile((xA - xB), [lower, upper]),
                        np.percentile((sig2A - sig2B), [lower, upper]))
            else:
                return (np.percentile((xB - xA), [lower, upper]),
                        np.percentile((sig2B - sig2A), [lower, upper])), (
                        np.percentile((xA - xB), [lower, upper]),
                        np.percentile((sig2A - sig2B), [lower, upper]))
        else:
            muA = self.modelA.loc_posterior
            aA = self.modelA.shape_posterior
            bA = self.modelA.scale_posterior

            muB = self.modelB.loc_posterior
            aB = self.modelB.shape_posterior
            bB = self.modelB.scale_posterior

            sig_mean_A, sig_var_A = self.modelA.std()
            sig_mean_B, sig_var_B = self.modelB.std()

            mu_mean = muB - muA
            sigma_mean = np.hypot(sig_mean_A, sig_mean_B)

            mu_var = bB / (aB - 1) - bA / (aA - 1)
            sigma_var = np.hypot(sig_var_A, sig_var_B)

            if variant == "A":
                return (stats.norm(mu_mean, sigma_mean).ppf([lower, upper]),
                        stats.norm(mu_var, sigma_var).ppf([lower, upper]))
            elif variant == "B":
                return (stats.norm(-mu_mean, sigma_mean).ppf([lower, upper]),
                        stats.norm(-mu_var, sigma_var).ppf([lower, upper]))
            else:
                return (stats.norm(mu_mean, sigma_mean).ppf([lower, upper]),
                        stats.norm(mu_var, sigma_var).ppf([lower, upper])), (
                        stats.norm(-mu_mean, sigma_mean).ppf([lower, upper]),
                        stats.norm(-mu_var, sigma_var).ppf([lower, upper]))

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
            The method of computation. Options are "asymptotic", "exact" and
            "MC".

        variant : str (default="A")
            The chosen variant. Options are "A", "B", "all".

        interval_length : float (default=0.9)
            Compute ``interval_length``\% credible interval. This is a value in
            [0, 1].

        Returns
        -------
        expected_loss_relative_ci : tuple of floats

        Notes
        -----
        Method "exact" uses the normal approximation of the Student's
        t-distribution for the expected loss of the mean.
        """
        check_ab_method(method=method,
                        method_options=("asymptotic", "exact", "MC"),
                        variant=variant, interval_length=interval_length)

        lower = (1 - interval_length) / 2
        upper = (1 + interval_length) / 2

        if method == "MC":
            data_A = self.modelA.rvs(self.simulations, self.random_state)
            data_B = self.modelB.rvs(self.simulations, self.random_state)

            xA, sig2A = data_A[:, 0], data_A[:, 1]
            xB, sig2B = data_B[:, 0], data_B[:, 1]

            lower *= 100.0
            upper *= 100.0

            if variant == "A":
                return (np.percentile((xB - xA) / xA, [lower, upper]),
                        np.percentile((sig2B - sig2A) / sig2A, [lower, upper]))
            elif variant == "B":
                return (np.percentile((xA - xB) / xB, [lower, upper]),
                        np.percentile((sig2A - sig2B) / sig2B, [lower, upper]))
            else:
                return (
                    np.percentile((xB - xA) / xA, [lower, upper]),
                    np.percentile((sig2B - sig2A) / sig2A, [lower, upper])), (
                    np.percentile((xA - xB) / xB, [lower, upper]),
                    np.percentile((sig2A - sig2B) / sig2B, [lower, upper]))
        else:
            # compute asymptotic
            muA = self.modelA.loc_posterior
            aA = self.modelA.shape_posterior
            bA = self.modelA.scale_posterior

            muB = self.modelB.loc_posterior
            aB = self.modelB.shape_posterior
            bB = self.modelB.scale_posterior

            sig2A = self.modelA.var()[0]
            sig2B = self.modelB.var()[0]

            if variant == "A":
                # mean using asymptotic normal approximation
                mu = muB / muA + sig2A * muB / muA ** 3
                var = sig2A * muB / muA ** 4 + sig2B / muA ** 2
                sigma = np.sqrt(var)
                dist = stats.norm(mu, sigma)
                ppfl_mean, ppfu_mean = dist.ppf([lower, upper])

                # variance
                mu = bB / bA * aA / (aB - 1)
                var = aA * (aA + aB - 1) / (aB - 2) / (aB - 1)**2
                var *= (bB / bA) ** 2
                sigma = np.sqrt(var)

                dist = stats.norm(mu, sigma)
                ppfl_var, ppfu_var = dist.ppf([lower, upper])

                if method == "exact":
                    ppfl_var = optimize.newton(
                        func=func_ppf, x0=ppfl_var, args=(
                            aA, bA, aB, bB, lower), maxiter=100)

                    ppfu_var = optimize.newton(
                        func=func_ppf, x0=ppfu_var, args=(
                            aA, bA, aB, bB, upper), maxiter=100)

                return ([ppfl_mean - 1, ppfu_mean - 1],
                        [ppfl_var - 1, ppfu_var - 1])

            elif variant == "B":
                # mean using asymptotic normal approximation
                mu = muA / muB + sig2B * muA / muB ** 3
                var = sig2B * muA / muB ** 4 + sig2A / muB ** 2
                sigma = np.sqrt(var)
                dist = stats.norm(mu, sigma)
                ppfl_mean, ppfu_mean = dist.ppf([lower, upper])

                # variance
                mu = bA / bB * aB / (aA - 1)
                var = aB * (aB + aA - 1) / (aA - 2) / (aA - 1)**2
                var *= (bA / bB) ** 2
                sigma = np.sqrt(var)

                dist = stats.norm(mu, sigma)
                ppfl_var, ppfu_var = dist.ppf([lower, upper])

                if method == "exact":
                    ppfl_var = optimize.newton(
                        func=func_ppf, x0=ppfl_var, args=(
                            aB, bB, aA, bA, lower), maxiter=100)

                    ppfu_var = optimize.newton(
                        func=func_ppf, x0=ppfu_var, args=(
                            aB, bB, aA, bA, upper), maxiter=100)

                return ([ppfl_mean - 1, ppfu_mean - 1],
                        [ppfl_var - 1, ppfu_var - 1])
            else:
                return (
                    self.expected_loss_relative_ci(
                        method=method,
                        variant="A", interval_length=interval_length)), (
                    self.expected_loss_relative_ci(
                        method=method,
                        variant="B", interval_length=interval_length))


class NormalInverseGammaMVTest(BayesMVTest):
    """
    Bayesian Multivariate testing with prior normal-inverse-gamma distribution.

    Parameters
    ----------
    models : object
        The normal-inverse-gamma models.

    simulations : int or None (default=1000000)
        Number of Monte Carlo simulations.

    random_state : int or None (default=None)
        The seed used by the random number generator.
    """
    def __init__(self, models, simulations=1000000, random_state=None,
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
        probability : tuple of floats

        Notes
        -----
        Method "exact" uses the normal approximation of the Student's
        t-distribution for the error probability of the mean when the number
        of degrees of freedom is large. For small values, numerical
        intergration is used.
        """
        check_mv_method(method=method, method_options=("exact", "MC"),
                        control=control, variant=variant,
                        variants=self.models.keys(), lift=lift)

        model_control = self.models[control]
        model_variant = self.models[variant]

        if method == "exact":
            mu0 = model_control.loc_posterior
            la0 = model_control.variance_scale_posterior
            a0 = model_control.shape_posterior
            b0 = model_control.scale_posterior

            mu1 = model_variant.loc_posterior
            la1 = model_variant.variance_scale_posterior
            a1 = model_variant.shape_posterior
            b1 = model_variant.scale_posterior

            if min(a0, a1) > 50:
                # mean using normal approximation
                sig0 = model_control.std()[0]
                sig1 = model_variant.std()[0]

                prob_mean = special.ndtr((mu1 - mu0) / np.hypot(sig0, sig1))
            else:
                s0 = np.sqrt(b0 / a0 / la0)
                s1 = np.sqrt(b1 / a1 / la1)

                prob_mean = integrate.quad(
                    func=func_ab_prob, a=-np.inf, b=np.inf, args=(
                        mu1, s1, 2 * a1, mu0, s0, 2 * a0))[0]

            # variance
            p = b1 / (b0 + b1)
            prob_var = special.betainc(a1, a0, p)
            return prob_mean, prob_var
        else:
            data_0 = model_control.rvs(self.simulations, self.random_state)
            data_1 = model_variant.rvs(self.simulations, self.random_state)

            x0, sig20 = data_0[:, 0], data_0[:, 1]
            x1, sig21 = data_1[:, 0], data_1[:, 1]

            return (x1 > x0 + lift).mean(), (sig21 > sig20 + lift).mean()

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
        probability_vs_all : tuple of floats
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

            return (xvariant > maxall + lift).mean(axis=0)
        else:
            # prepare parameters
            variant_params = [(self.models[v].loc_posterior,
                              self.models[v].variance_scale_posterior,
                              self.models[v].shape_posterior,
                              self.models[v].scale_posterior)
                              for v in variants]

            mu = self.models[variant].loc_posterior
            la = self.models[variant].variance_scale_posterior
            a = self.models[variant].shape_posterior
            b = self.models[variant].scale_posterior

            v = 2 * a
            s = np.sqrt(b / a / la)

            if method == "quad":
                # mean
                min_t, max_t = stats.t(df=v, loc=mu, scale=s).ppf(
                    [0.00000001, 0.99999999])

                prob_mean = integrate.quad(
                    func=func_mv_prob_mean, a=min_t, b=max_t, args=(
                        mu, s, v, variant_params))[0]

                # variance
                n = stats.invgamma(a=a, scale=b).ppf(0.99999999)
                prob_var = integrate.quad(
                    func=func_mv_prob_var, a=0, b=n, args=(
                        a, b, variant_params))[0]

                return prob_mean, prob_var
            else:
                r = np.arange(mlhs_samples)
                np.random.shuffle(r)
                r = (r - 0.5) / mlhs_samples
                r = r[r >= 0]

                # mean
                x = stats.t(df=2 * a, loc=mu, scale=np.sqrt(b / a / la)).ppf(r)

                prob_mean = np.nanmean(
                    np.prod([special.stdtr(2*a, (x - mu) / np.sqrt(b / a / la))
                            for mu, la, a, b in variant_params], axis=0))

                # variance
                x = stats.invgamma(a=a, scale=b).ppf(r)
                prob_var = np.nanmean(np.prod([special.gammaincc(a, b / x)
                                      for _, _, a, b in variant_params],
                                      axis=0))

                return prob_mean, prob_var

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
        expected_loss : tuple of floats

        Notes
        -----
        Method "exact" uses the normal approximation of the Student's
        t-distribution for the expected loss of the mean when the number
        of degrees of freedom is large. For small values, numerical
        intergration is used.
        """
        check_mv_method(method=method, method_options=("exact", "MC"),
                        control=control, variant=variant,
                        variants=self.models.keys(), lift=lift)

        model_control = self.models[control]
        model_variant = self.models[variant]

        if method == "exact":
            mu0 = model_control.loc_posterior
            la0 = model_control.variance_scale_posterior
            a0 = model_control.shape_posterior
            b0 = model_control.scale_posterior

            mu1 = model_variant.loc_posterior
            la1 = model_variant.variance_scale_posterior
            a1 = model_variant.shape_posterior
            b1 = model_variant.scale_posterior

            if min(a0, a1) > 50 and max(a0, a1) <= 1:
                # mean using normal approximation
                sig0 = model_control.std()[0]
                sig1 = model_variant.std()[0]

                u = mu1 - mu0
                s = np.hypot(sig0, sig1)
                t = s * np.exp(-0.5 * (u / s) ** 2) / np.sqrt(2 * np.pi)
                el_mean = t - u * special.ndtr(-u / s)
            else:
                # numerical integration
                s0 = np.sqrt(b0 / a0 / la0)
                s1 = np.sqrt(b1 / a1 / la1)

                el_mean = integrate.quad(
                    func=func_ab_el, a=-np.inf, b=np.inf, args=(
                        mu0, s0, 2*a0, mu1, s1, 2*a1))[0]

            # variance
            if min(a0, a1) > 1:
                t0 = b0 / (a0 - 1) * special.betainc(
                    a0 - 1, a1, b0 / (b0 + b1))
                t1 = b1 / (a1 - 1) * special.betainc(
                    a0, a1 - 1, b0 / (b0 + b1))
                el_var = t0 - t1
            else:
                el_var = np.nan
            return el_mean, el_var
        else:
            data_0 = model_control.rvs(self.simulations, self.random_state)
            data_1 = model_variant.rvs(self.simulations, self.random_state)

            x0, sig20 = data_0[:, 0], data_0[:, 1]
            x1, sig21 = data_1[:, 0], data_1[:, 1]

            return (np.maximum(x0 - x1 - lift, 0).mean(),
                    np.maximum(sig20 - sig21 - lift, 0).mean())

    def expected_loss_ci(self, method="MC", control="A", variant="B",
                         interval_length=0.9):
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

        Returns
        -------
        expected_loss_ci : tuple of floats
        """
        check_mv_method(method=method, method_options=("MC", "asymptotic"),
                        control=control, variant=variant,
                        variants=self.models.keys(),
                        interval_length=interval_length)

        # check interval length
        lower = (1 - interval_length) / 2
        upper = (1 + interval_length) / 2

        model_control = self.models[control]
        model_variant = self.models[variant]

        if method == "MC":
            data_0 = model_control.rvs(self.simulations, self.random_state)
            data_1 = model_variant.rvs(self.simulations, self.random_state)

            x0, sig20 = data_0[:, 0], data_0[:, 1]
            x1, sig21 = data_1[:, 0], data_1[:, 1]

            lower *= 100.0
            upper *= 100.0

            return (np.percentile((x0 - x1), [lower, upper]),
                    np.percentile((sig20 - sig21), [lower, upper]))
        else:
            mu0 = model_control.loc_posterior
            a0 = model_control.shape_posterior
            b0 = model_control.scale_posterior

            mu1 = model_variant.loc_posterior
            a1 = model_variant.shape_posterior
            b1 = model_variant.scale_posterior

            sig_mean_0, sig_var_0 = model_control.std()
            sig_mean_1, sig_var_1 = model_variant.std()

            mu_mean = mu1 - mu0
            sigma_mean = np.hypot(sig_mean_0, sig_mean_1)

            mu_var = b1 / (a1 - 1) - b0 / (a0 - 1)
            sigma_var = np.hypot(sig_var_0, sig_var_1)

            return (stats.norm(-mu_mean, sigma_mean).ppf([lower, upper]),
                    stats.norm(-mu_var, sigma_var).ppf([lower, upper]))

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
        expected_loss_relative : tuple of floats

        Notes
        -----
        Method "exact" uses an approximation of :math:`E[1/X]` where
        :math:`X` follows a Student's t-distribution.
        """
        check_mv_method(method=method, method_options=("exact", "MC"),
                        control=control, variant=variant,
                        variants=self.models.keys())

        model_control = self.models[control]
        model_variant = self.models[variant]

        if method == "exact":
            mu0 = model_control.loc_posterior
            a0 = model_control.shape_posterior
            b0 = model_control.scale_posterior

            mu1 = model_variant.loc_posterior
            a1 = model_variant.shape_posterior
            b1 = model_variant.scale_posterior

            sig21 = model_variant.var()[0]

            # mean using asymptotic normal approximation
            elr_mean = mu0 / mu1 + sig21 * mu0 / mu1 ** 3 - 1

            # variance
            elr_var = b0 / b1 * a1 / (a0 - 1) - 1

            return elr_mean, elr_var
        else:
            data_0 = model_control.rvs(self.simulations, self.random_state)
            data_1 = model_variant.rvs(self.simulations, self.random_state)

            x0, sig20 = data_0[:, 0], data_0[:, 1]
            x1, sig21 = data_1[:, 0], data_1[:, 1]

            return ((x0 - x1) / x1).mean(), ((sig20 - sig21) / sig21).mean()

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
        expected_loss_relative_vs_all : tuple of floats
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

            return (maxall / xvariant).mean(axis=0) - 1
        else:
            # prepare parameters
            variant_params = [(self.models[v].loc_posterior,
                              self.models[v].variance_scale_posterior,
                              self.models[v].shape_posterior,
                              self.models[v].scale_posterior)
                              for v in variants]

            mu = self.models[variant].loc_posterior
            la = self.models[variant].variance_scale_posterior
            a = self.models[variant].shape_posterior
            b = self.models[variant].scale_posterior

            if method == "quad":
                max_ig = np.max([stats.invgamma(
                    a=self.models[v].shape_posterior,
                    scale=self.models[v].scale_posterior).ppf(0.99999999)
                    for v in variants])

                t_ppfs = [stats.t(
                    df=2*self.models[v].shape_posterior,
                    loc=self.models[v].loc_posterior,
                    scale=np.sqrt(
                        self.models[v].scale_posterior /
                        self.models[v].shape_posterior /
                        self.models[v].variance_scale_posterior)).ppf(
                    [0.00000001, 0.99999999]) for v in variants]

                min_t = np.min([q[0] for q in t_ppfs])
                max_t = np.max([q[1] for q in t_ppfs])

                # mean
                e_max = integrate.quad(func=func_mv_elr_mean, a=min_t,
                                       b=max_t, args=(variant_params))[0]

                e_inv_x = (1 + self.models[variant].var()[0] / mu ** 2) / mu

                elr_mean = e_max * e_inv_x - 1

                # variance
                e_max = integrate.quad(func=func_mv_elr_var, a=0, b=max_ig,
                                       args=(variant_params))[0]
                e_inv_x = a / b

                elr_variance = e_max * e_inv_x - 1

                return elr_mean, elr_variance
            else:
                r = np.arange(mlhs_samples)
                np.random.shuffle(r)
                r = (r - 0.5) / mlhs_samples
                r = r[r >= 0][..., np.newaxis]

                n = len(variant_params)
                variant_params.append((mu, la, a, b))
                uu, ll, aa, bb = map(np.array, zip(*variant_params))
                vv = 2 * aa
                ss = np.sqrt(bb / aa / ll)

                # mean
                xx = stats.t(df=vv, loc=uu, scale=ss).ppf(r)
                xr = (1. / xx[:, -1]).mean()

                elr_mean = np.sum(
                    xx[:, :-1].T * [np.prod([
                        special.stdtr(vv[j], (xx[:, i] - uu[j]) / ss[j])
                        for j in range(n) if j != i],
                        axis=0) for i in range(n)], axis=0).mean() * xr - 1

                # variance
                xx = stats.invgamma(a=aa, scale=bb).ppf(r)
                xr = (1. / xx[:, -1]).mean()

                elr_var = np.sum(
                    xx[:, :-1].T * [np.prod([
                        special.gammaincc(aa[j], bb[j] / xx[:, i])
                        for j in range(n) if j != i],
                        axis=0) for i in range(n)], axis=0).mean() * xr - 1

                return elr_mean, elr_var

    def expected_loss_relative_ci(self, method="MC", control="A", variant="B",
                                  interval_length=0.9):
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

        Returns
        -------
        expected_loss_relative_ci : tuple of floats
        """
        check_mv_method(method=method,
                        method_options=("asymptotic", "exact", "MC"),
                        control=control, variant=variant,
                        variants=self.models.keys(),
                        interval_length=interval_length)

        lower = (1 - interval_length) / 2
        upper = (1 + interval_length) / 2

        model_control = self.models[control]
        model_variant = self.models[variant]

        if method == "MC":
            data_0 = model_control.rvs(self.simulations, self.random_state)
            data_1 = model_variant.rvs(self.simulations, self.random_state)

            x0, sig20 = data_0[:, 0], data_0[:, 1]
            x1, sig21 = data_1[:, 0], data_1[:, 1]

            lower *= 100.0
            upper *= 100.0

            return (np.percentile((x0 - x1) / x1, [lower, upper]),
                    np.percentile((sig20 - sig21) / sig21, [lower, upper]))
        else:
            mu0 = model_control.loc_posterior
            a0 = model_control.shape_posterior
            b0 = model_control.scale_posterior

            mu1 = model_variant.loc_posterior
            a1 = model_variant.shape_posterior
            b1 = model_variant.scale_posterior

            sig20 = model_control.var()[0]
            sig21 = model_variant.var()[0]

            # mean using asymptotic normal approximation
            mu = mu0 / mu1 + sig21 * mu0 / mu1 ** 3
            var = sig21 * mu0 / mu1 ** 4 + sig20 / mu1 ** 2
            sigma = np.sqrt(var)
            dist = stats.norm(mu, sigma)
            ppfl_mean, ppfu_mean = dist.ppf([lower, upper])

            # variance
            mu = b0 / b1 * a1 / (a0 - 1)
            var = a1 * (a1 + a0 - 1) / (a0 - 2) / (a0 - 1)**2
            var *= (b0 / b1) ** 2
            sigma = np.sqrt(var)

            dist = stats.norm(mu, sigma)
            ppfl_var, ppfu_var = dist.ppf([lower, upper])

            if method == "exact":
                ppfl_var = optimize.newton(func=func_ppf, x0=ppfl_var, args=(
                    a1, b1, a0, b0, lower), maxiter=100)

                ppfu_var = optimize.newton(func=func_ppf, x0=ppfu_var, args=(
                    a1, b1, a0, b0, upper), maxiter=100)

            return ([ppfl_mean - 1, ppfu_mean - 1],
                    [ppfl_var - 1, ppfu_var - 1])

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
        expected_loss_vs_all : tuple of floats
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

            return np.maximum(maxall - xvariant - lift, 0).mean(axis=0)
        else:
            max_ig = np.max([stats.invgamma(
                a=self.models[v].shape_posterior,
                scale=self.models[v].scale_posterior).ppf(0.99999999)
                for v in variants])

            t_ppfs = [stats.t(
                df=2*self.models[v].shape_posterior,
                loc=self.models[v].loc_posterior,
                scale=np.sqrt(
                    self.models[v].scale_posterior /
                    self.models[v].shape_posterior /
                    self.models[v].variance_scale_posterior)).ppf(
                [0.00000001, 0.99999999]) for v in variants]

            min_t = np.min([q[0] for q in t_ppfs])
            max_t = np.max([q[1] for q in t_ppfs])

            # exclude variant
            variants.remove(variant)

            # prepare parameters
            variant_params = [(self.models[v].loc_posterior,
                              self.models[v].variance_scale_posterior,
                              self.models[v].shape_posterior,
                              self.models[v].scale_posterior)
                              for v in variants]

            mu = self.models[variant].loc_posterior
            la = self.models[variant].variance_scale_posterior
            a = self.models[variant].shape_posterior
            b = self.models[variant].scale_posterior

            v = 2 * a
            s = np.sqrt(b / a / la)

            if method == "quad":
                # mean
                el_mean = integrate.quad(
                    func=func_mv_el_mean, a=min_t, b=max_t, args=(
                        mu, s, v, variant_params))[0]

                # variance
                el_var = integrate.quad(
                    func=func_mv_el_var, a=0, b=max_ig, args=(
                        a, b, variant_params))[0]

                return el_mean, el_var
            else:
                r = np.arange(mlhs_samples)
                np.random.shuffle(r)
                r = (r - 0.5) / mlhs_samples
                r = r[r >= 0]

                # mean
                x = np.array([optimize.brentq(
                    f=func_mv_student_ppf, args=(variant_params, p), a=min_t,
                    b=max_t, xtol=1e-4, rtol=1e-4) for p in r])

                # compute second integral
                t = (x - mu) / s
                c = s * np.exp(0.5 * (1 - v) * np.log(1 + t ** 2 / v)
                               + 0.5 * np.log(v)
                               - special.betaln(v * 0.5, 0.5)) / (1 - v)

                el_mean = np.nanmean((x - mu) * special.stdtr(v, t) - c)

                # variance
                x = np.array([optimize.brentq(
                    f=func_mv_inverse_gamma_ppf, args=(variant_params, p), a=0,
                    b=max_ig, xtol=1e-4, rtol=1e-4) for p in r])

                p = x * special.gammaincc(a, b / x)
                q = b / (a - 1) * special.gammaincc(a - 1, b / x)
                el_var = np.nanmean(p - q)

                return el_mean, el_var
