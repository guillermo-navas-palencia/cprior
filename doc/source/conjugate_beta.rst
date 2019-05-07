Beta distribution
=================

The probability density function of the beta distribution :math:`\mathcal{B}(\alpha, \beta)` with two shape parameters :math:`\alpha, \beta > 0`, for :math:`x \in [0, 1]`, is defined by

.. math::

   f(x; \alpha, \beta) = \frac{x^{\alpha - 1} (1-x)^{\beta - 1}}{B(\alpha, \beta)},

and the cumulative distribution function is

.. math::

   F(x; \alpha, \beta) = I_x(\alpha, \beta),

where :math:`B(\alpha, \beta)` is the beta function and :math:`I_x(\alpha, \beta)` is the regularized incomplete beta function. The expected value and variance are as follows

.. math::
   \mathrm{E}[X] = \frac{\alpha}{\alpha + \beta}, \quad \mathrm{Var}[X] = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta - 1)}.

In Bayesian inference, the beta distribution is the conjugate prior probability distribution of the Bernoulli, binomial, negative binomial and geometric distribution. The beta distribution is a suitable model for the random behaviour of percentages and proportions.  

.. autoclass:: cprior.cdist.BetaModel
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.cdist.BetaABTest
   :members:
   :inherited-members:
   :show-inheritance:
