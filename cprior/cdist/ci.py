"""
Confidence/credible intervals (CI) methods.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numbers

import numpy as np

from scipy import optimize


def ci_interval(x, interval_length=0.9, method="ETI"):
    r"""
    Compute confidence/credible intervals (CI).

    Parameters
    ----------
    x : array-like, shape = (n_samples)

    interval_length : float (default=0.9)
        Compute ``interval_length``\% credible interval. This is a value in
        [0, 1].

    method : str (default="ETI")
        Method to compute credible intervals. Supported methods are Highest
        Density interval (``method="HDI"``) and Equal-tailed interval
        (``method="ETI"``).

    Returns
    -------
    ci_interval : numpy.ndarray
        The lower and upper limit of the interval.

    Example
    -------
    >>> from scipy import stats
    >>> from cprior.cdist import ci_interval
    >>> x = stats.norm.rvs(size=int(1e6), random_state=42)
    >>> ci_interval(x=x, interval_length=0.95, method="ETI")
    array([-1.96315029,  1.95842544])
    >>> ci_interval(x=x, interval_length=0.95, method="HDI")
    array([-1.95282024,  1.9679026 ])
    """
    if method not in ("ETI", "HDI"):
        raise ValueError("method {} is not supported. Use 'ETI' or 'HDI'"
                         .format(method))

    invalid_length = (interval_length < 0 or interval_length > 1)

    if not isinstance(interval_length, numbers.Number) or invalid_length:
        raise ValueError("Interval length must a value in [0, 1]; got "
                         "interval_length={}.".format(interval_length))

    if method == "ETI":
        lower = 100 * (1 - interval_length) / 2
        upper = 100 * (1 + interval_length) / 2

        return np.percentile(x, [lower, upper])
    else:
        n = len(x)
        xsorted = np.sort(x)
        n_included = int(np.ceil(interval_length * n))
        n_ci = n - n_included
        ci = xsorted[n_included:] - xsorted[:n_ci]
        j = np.argmin(ci)
        hdi_min = xsorted[j]
        hdi_max = xsorted[j + n_included]

        return np.array([hdi_min, hdi_max])


def _ci_hdi_exact(f, x0, interval_length, bounds):
    def func(x):
        return x[3] + x[2] + x[1] - x[0]

    def obj_f(x):
        return f.pdf(x[1]) - f.pdf(x[0])

    def obj_F(x):
        return f.cdf(x[1]) - f.cdf(x[0])

    epsilon = 1e-6

    cons = (
        {'type': 'ineq', 'fun': lambda x: x[1] - x[0] - epsilon},
        {'type': 'ineq', 'fun': lambda x: -x[2] + obj_f(x)},
        {'type': 'ineq', 'fun': lambda x: x[2] + obj_f(x)},
        {'type': 'ineq', 'fun': lambda x: -x[3] + obj_F(x) - interval_length},
        {'type': 'ineq', 'fun': lambda x: x[3] + obj_F(x) - interval_length}
    )

    res = optimize.minimize(func, (*x0, 0, 0), method="SLSQP",
                            constraints=cons, bounds=[*bounds, (0, 1), (0, 1)])

    return res


def ci_interval_exact(dist, interval_length=0.9, method="ETI", bounds=None):
    r"""
    Compute exact confidence/credible intervals (CI).

    Parameters
    ----------
    dist : object
        A class representing a probability distribution with methods ``pdf``,
        ``cdf`` and ``ppf``.

    interval_length : float (default=0.9)
        Compute ``interval_length``\% credible interval. This is a value in
        [0, 1].

    method : str (default="ETI")
        Method to compute credible intervals. Supported methods are Highest
        Density interval (``method="HDI"``) and Equal-tailed interval
        (``method="ETI"``).

    bounds : list or None (default=None)
        Sequence of ``(min, max)`` pairs for lower and upper limit of the
        interval. None is used to specify no bound.

    Returns
    -------
    ci_interval : numpy.ndarray
        The lower and upper limit of the interval.

    Example
    -------
    >>> from scipy import stats
    >>> from cprior.cdist import ci_interval
    >>> dist = stats.beta(4, 10)
    >>> ci_interval_exact(dist=dist, interval_length=0.9, method="ETI")
    array([0.11266578, 0.49464973])
    >>> bounds = [(0, 1), (0, 1)]
    >>> ci_interval_exact(dist=dist, interval_length=0.9, method="HDI", bounds=bounds)
    array([0.09439576, 0.46944915])
    """
    if method not in ("ETI", "HDI"):
        raise ValueError("method {} is not supported. Use 'ETI' or 'HDI'"
                         .format(method))

    if not isinstance(interval_length, numbers.Number) or (
            interval_length < 0 or interval_length > 1):
        raise ValueError("Interval length must a value in [0, 1]; got "
                         "interval_length={}.".format(interval_length))

    if not all(hasattr(dist, attr) for attr in ["pdf", "cdf", "ppf"]):
        raise TypeError("dist must have methods 'pdf', 'cdf' and 'ppf'.")

    lower = (1 - interval_length) / 2
    upper = (1 + interval_length) / 2

    if method == "ETI":
        return dist.ppf([lower, upper])
    else:
        if bounds is None:
            bounds = [(-np.inf, np.inf), (-np.inf, np.inf)]

        x0 = dist.ppf([lower, upper])
        res = _ci_hdi_exact(dist, x0, interval_length, bounds)

        if res.success:
            return res.x[:2]
        else:
            return np.array([np.nan, np.nan])
