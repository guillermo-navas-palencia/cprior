Pareto distribution
===================

The probability density function of the Pareto distribution :math:`\mathcal{PA}(\alpha, \beta)` with shape parameter :math:`\alpha > 0` and
scale parameter :math:`\beta > 0`, for :math:`x \in [\beta, \infty)`, is given by

.. math::

   f(x; \alpha, \beta) = \frac{\alpha \beta^{\alpha}}{x^{\alpha + 1}},

and the cumulative distribution function is

.. math::

   F(x; \alpha, \beta) = 1 - \left(\frac{\beta}{x}\right)^{\alpha}.

The expected value and variance are as follows

.. math::

   \mathrm{E}[X] = \frac{\alpha \beta}{\alpha - 1}, \quad \mathrm{Var}[X] = \frac{\beta^2 \alpha}{(\alpha - 1)^2 (\alpha - 2)}.

The Pareto distribution is used as a conjugate prior distribution for the uniform distribution.

.. autoclass:: cprior.cdist.ParetoModel
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.cdist.ParetoABTest
   :members:
   :inherited-members:
   :show-inheritance: