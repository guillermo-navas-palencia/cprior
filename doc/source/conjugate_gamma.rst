Gamma distribution
==================

The probability density function of the gamma distribution :math:`\mathcal{G}(\alpha, \beta)` with shape parameter :math:`\alpha > 0` and rate parameter :math:`\beta > 0`, for :math:`x\in (0, \infty)`, is given by

.. math::

   f(x; \alpha, \beta) = \frac{\beta^{\alpha} x^{\alpha-1} e^{-\beta x}}{\Gamma(a)},

and the cumulative distribution function is

.. math::

   F(x; \alpha, \beta) = P(\alpha, \beta x),

where :math:`P(\alpha, \beta x)` is the regularized lower incomplete gamma function. Using this parametrization of the gamma distribution, the expected value and variance are

.. math::

   \mathrm{E}[X] = \frac{\alpha}{\beta}, \quad \mathrm{Var}[X] = \frac{\alpha}{\beta^2}.

This parametrization is commonly used in Bayesian statistics, where the gamma function is used as a conjugate prior distribution for various distribution such as the exponential, Pareto and Poisson distribution.

.. autoclass:: cprior.cdist.GammaModel
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.cdist.GammaABTest
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.cdist.GammaMVTest
   :members:
   :inherited-members:
   :show-inheritance:
